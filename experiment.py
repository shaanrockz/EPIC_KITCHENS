from config import config
import torchvision
import transforms
from torch.utils.data import DataLoader
from dataset import EPIC
from custom_resnet import EPICModel
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from datetime import datetime
from torch.nn.utils import clip_grad_norm
import json
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
import random
from min_norm_solver import MinNormSolver, gradient_normalizers

class Experiment:
    def __init__(
        self,
        cfg,
        model,
        loss=None,
        optimizer=None,
        use_cuda=None,
        data_train=None,
        data_val=None,
        debugging=False,
    ):
        self.cfg = cfg
        self.use_cuda = use_cuda
        self.data_train = data_train
        self.data_val = data_val
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.load_model = self.cfg.load_model
        self.debugging = debugging

        if self.cfg.algo == "IRM":
            if self.use_cuda:
                self.dummy_w = torch.Tensor([1.0]).cuda()
            else:
                self.dummy_w = torch.Tensor([1.0])

            self.dummy_w.requires_grad = True
        
        if self.cfg.algo == "MTGA":
            self.minNormSolver = MinNormSolver()

    def adjust_learning_rate(self, optimizer, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = self.cfg.lr * decay
        decay = self.cfg.weight_decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * param_group["lr_mult"]
            param_group["weight_decay"] = decay * param_group["decay_mult"]

    def compute_irm_penalty(self, loss):

        penalty = grad(loss[0::2].mean(), self.dummy_w, create_graph=True)[0]
        penalty *= grad(loss[1::2].mean(), self.dummy_w, create_graph=True)[0]

        return penalty.sum()

    def uniqueSampling(self, arr, samplingSize):
        available_indices = [i for i in range(len(arr)) if arr[i] > 1]
        if len(available_indices) >= samplingSize:
            return available_indices

        available_indices = [i for i in range(len(arr)) if arr[i] > 0]
        if len(available_indices) >= samplingSize:
            return available_indices
        else:
            return []

    class Node(dict):
        def __init__(self):
            super().__init__()
            self["version"] = "0.1"
            self["challenge"] = "action_recognition"
            self["results"] = dict()

        def add_node(self, uid, verb, noun):
            tmp = {"verb": verb, "noun": noun}
            self["results"][str(int(uid))] = tmp

        def add(self, uid, verb, noun):
            verb_dict = dict()
            for verb_id in range(125):
                verb_dict[str(int(verb_id))] = verb[verb_id]
            noun_dict = dict()
            for noun_id in range(352):
                noun_dict[str(int(noun_id))] = noun[noun_id]
            self.add_node(uid, verb_dict, noun_dict)

    def _model_test(self, sample_batch):
        # Fetch data and annotation from the batch
        inputs = sample_batch["data"]
        uids = sample_batch["uid"]

        # If GPU available convert tensors to cuda tensors
        if self.use_cuda:
            inputs = inputs.cuda()

        tmp = self.model(inputs)
        output_verb = tmp["out_verb"].detach().cpu().numpy()
        output_noun = tmp["out_noun"].detach().cpu().numpy()

        return {
            "out_verb": output_verb,
            "out_noun": output_noun,
            "uids": uids,
        }
    
    def _model_feature(self, sample_batch):
        # Fetch data and annotation from the batch
        inputs = sample_batch["data"]

        # If GPU available convert tensors to cuda tensors
        if self.use_cuda:
            inputs = inputs.cuda()

        tmp = self.model(inputs)
        out_feat = tmp["out_shared"].detach().cpu().numpy()
        return out_feat


    def _model_output(self, sample_batch):
        # Fetch data and annotation from the batch
        inputs = sample_batch["data"]
        verb_anno = sample_batch["verb_id"]
        noun_anno = sample_batch["noun_id"]

        # If GPU available convert tensors to cuda tensors
        if self.use_cuda:
            inputs = inputs.cuda()
            verb_anno = verb_anno.cuda()
            noun_anno = noun_anno.cuda()

        tmp = self.model(inputs)
        output_verb = tmp["out_verb"]
        output_noun = tmp["out_noun"]

        if self.cfg.algo == "IRM":
            output_verb *= self.dummy_w
            output_noun *= self.dummy_w

        # Loss Evaluation
        loss_verb = self.loss(output_verb, verb_anno.view(-1))
        loss_noun = self.loss(output_noun, noun_anno.view(-1))

        # Transformation of loss for equal weightage
        if self.cfg.algo == "IRM":
            loss = 0.5 * (loss_verb.mean() + loss_noun.mean())
        else:
            loss = 0.5 * (loss_verb + loss_noun)

        return {
            "out_verb": output_verb,
            "out_noun": output_noun,
            "loss_erm": loss,
            "loss_verb": loss_verb,
            "loss_noun": loss_noun,
            "verb_anno": verb_anno,
            "noun_anno": noun_anno,
        }

    def _model_output_mtga(self, sample_batch):
        # Fetch data and annotation from the batch
        inputs = sample_batch["data"]
        verb_anno = sample_batch["verb_id"]
        noun_anno = sample_batch["noun_id"]

        # If GPU available convert tensors to cuda tensors
        if self.use_cuda:
            inputs = inputs.cuda()
            verb_anno = verb_anno.cuda()
            noun_anno = noun_anno.cuda()

        tmp = self.model(inputs)
        output_verb = tmp["out_verb"]
        output_noun = tmp["out_noun"]

        loss_data = {}
        # Loss Evaluation
        loss_verb = self.loss(output_verb, verb_anno.view(-1))
        loss_data[0] = loss_verb.data
        loss_noun = self.loss(output_noun, noun_anno.view(-1))
        loss_data[1] = loss_verb.data

        grads = {}

        self.optimizer.zero_grad()
        grads[0] = []
        loss_verb.backward(retain_graph=True)
        for param in self.model.base.parameters():
            if param.grad is not None:
                grads[0].append(param.grad.data.clone())
        
        self.optimizer.zero_grad()
        grads[1] = []
        loss_noun.backward(retain_graph=True)
        for param in self.model.base.parameters():
            if param.grad is not None:
                grads[1].append(param.grad.data.clone())

        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, "l2")
        for t in range(2):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        scale = {}
        vecs = [grads[t] for t in range(2)]
        sol, min_norm = self.minNormSolver.find_min_norm_element(vecs=vecs)
        for t in range(2):
            scale[t] = float(sol[t])

        tmp = self.model(inputs)
        output_verb = tmp["out_verb"]
        output_noun = tmp["out_noun"]

        # Loss Evaluation
        loss_verb = scale[0]*self.loss(output_verb, verb_anno.view(-1))
        loss_noun = scale[1]*self.loss(output_noun, noun_anno.view(-1))

        # Transformation of loss for equal weightage
        loss = loss_verb + loss_noun

        return {
            "out_verb": output_verb,
            "out_noun": output_noun,
            "loss_mtga": loss,
            "loss_verb": loss_verb,
            "loss_noun": loss_noun,
            "verb_anno": verb_anno,
            "noun_anno": noun_anno,
        }

    def euclidean_metric(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = (-((a - b)**2).sum(dim=2)).softmax(dim=1)
        return logits

    def _model_output_fsl(self, sample_batch_verb, sample_batch_noun):
        p = self.cfg.shot * self.cfg.way

        # Fetch data and annotation from the batch
        input_verb = sample_batch_verb["data"]
        input_noun = sample_batch_noun["data"]

        verb_anno_real = sample_batch_verb["verb_id"]
        noun_anno_real = sample_batch_noun["noun_id"]

        verb_anno = torch.arange(self.cfg.way).repeat(self.cfg.query)
        noun_anno = torch.arange(self.cfg.way).repeat(self.cfg.query)

        # If GPU available convert tensors to cuda tensors
        if self.use_cuda:
            input_verb = input_verb.cuda()
            input_noun = input_noun.cuda()

        tmp_verb = self.model(input_verb)["out_verb"]
        tmp_verb_shot = tmp_verb[:p]
        tmp_verb_query = tmp_verb[p:]
        #output_verb = self.euclidean_metric(tmp_verb_query, tmp_verb_shot) / self.cfg.temperature

        tmp_noun = self.model(input_noun)["out_noun"]
        tmp_noun_shot = tmp_noun[:p]
        tmp_noun_query = tmp_noun[p:]
        #output_noun = self.euclidean_metric(tmp_noun_query, tmp_noun_shot) / self.cfg.temperature

        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        prototypes_verb = tmp_verb_shot.reshape(self.cfg.way, self.cfg.shot, -1).mean(dim=1)
        prototypes_noun = tmp_noun_shot.reshape(self.cfg.way, self.cfg.shot, -1).mean(dim=1)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        n_x = tmp_verb_query.shape[0]
        n_y = prototypes_verb.shape[0]
        distances_verb = (
                tmp_verb_query.unsqueeze(1).expand(n_x, n_y, -1) -
                prototypes_verb.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)

        n_x = tmp_noun_query.shape[0]
        n_y = prototypes_noun.shape[0]
        distances_noun = (
                tmp_noun_query.unsqueeze(1).expand(n_x, n_y, -1) -
                prototypes_noun.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)

        # Calculate log p_{phi} (y = k | x)
        clipped_y_pred_verb = (-distances_verb).double()
        clipped_y_pred_noun = (-distances_noun).double()
        
        if self.use_cuda:
            verb_anno = verb_anno.cuda()
            noun_anno = noun_anno.cuda()
            verb_anno_real = verb_anno_real.cuda()
            noun_anno_real = noun_anno_real.cuda()

        # Loss Evaluation
        loss_verb = self.loss(clipped_y_pred_verb, verb_anno.view(-1)) + self.loss(tmp_verb_query, verb_anno_real.view(-1)[p:])
        loss_noun = self.loss(clipped_y_pred_noun, noun_anno.view(-1)) + self.loss(tmp_noun_query, noun_anno_real.view(-1)[p:])

        # Transformation of loss for equal weightage
        loss = 0.5 * (loss_verb + loss_noun)

        return {
            "loss_erm": loss,
            "loss_verb": loss_verb,
            "loss_noun": loss_noun,
            "verb_anno": verb_anno,
            "noun_anno": noun_anno,
        }

    def save_checkpoint(self, state, intermediate=False):
        if intermediate:
            torch.save(
                state,
                "Epoch_" + str(state["epoch"]) + "_" + self.cfg.checkpoint_filename,
            )
        else:
            torch.save(state, self.cfg.checkpoint_filename)

    def save_final_checkpoint(self, state):
        torch.save(state, self.cfg.checkpoint_filename_final)

    def train(self):
        iter_num = 0

        # Criteria parameter for saving the model (if better than best then save it, checking happens every 10 epoch)
        best_val_loss = np.inf
        epoch = 1
        random.seed(self.cfg.manualSeed)

        # Load Model Checkpoint
        if self.load_model:
            checkpoint = torch.load(self.cfg.checkpoint_filename_final)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"] + 1
            # best_val_loss = checkpoint["loss"]
            iter_num = checkpoint["iter_num"]
            self.summary_filename = checkpoint["summary_filename"]
            self.writer = SummaryWriter(
                log_dir="runs/" + self.summary_filename,
                flush_secs=30,
                filename_suffix=self.summary_filename,
            )

        self.optimizer.zero_grad()

        penalty_global_idx = 0
        # Looping over epoch
        while epoch <= self.cfg.epoch:

            penalty_global_idx += 1

            print("Epoch : " + str(epoch))

            if self.cfg.algo == "IRM":
                # Adjust learning rate for IRM training
                self.adjust_learning_rate(self.optimizer, epoch, [20, 35])
            elif self.cfg.algo == "ERM" or self.cfg.algo == "MTGA":
                # Adjust learning rate for ERM training
                self.adjust_learning_rate(self.optimizer, epoch, [22, 40])
            elif self.cfg.algo == "FSL":
                # Adjust learning rate for ERM training
                self.adjust_learning_rate(self.optimizer, epoch, [20, 40])

            losses = []

            # Train mode activation for model

            if self.cfg.algo == "ERM":
                self.model.train()
                # Enumerate over batches
                for idx, sample_batch in enumerate(self.data_train):
                    # if self.cfg.my_base_model:
                    out = self._model_output(sample_batch)
                    # else:
                    #     out = self._model_output_tsn(sample_batch)
                    loss = out["loss_erm"]
                    losses.append(loss.item())

                    # Zero gradients, perform a backward pass, and update the weights.
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Clipping Gradient
                    clip_grad_norm(self.model.parameters(), 20)

                    self.optimizer.step()

                    # For faster debugging
                    if self.debugging:
                        if idx == 20:
                            break
            
            elif self.cfg.algo == "MTGA":
                self.model.train()
                # Enumerate over batches
                for idx, sample_batch in enumerate(self.data_train):

                    # if self.cfg.my_base_model:
                    out = self._model_output_mtga(sample_batch)

                    loss = out["loss_mtga"]
                    losses.append(loss.item())

                    self.optimizer.zero_grad()

                    # perform a backward pass, and update the weights.
                    loss.backward()

                    # Clipping Gradient
                    clip_grad_norm(self.model.parameters(), 20)

                    self.optimizer.step()


                    # For faster debugging
                    if self.debugging:
                        if idx == 20:
                            break

            elif self.cfg.algo == "FSL":
                self.model.train()
                trainloader_verb = iter(self.data_train["verb"])
                trainloader_noun = iter(self.data_train["noun"])
                # Enumerate over batches
                idx = 0
                loop = True

                while loop:
                    self.optimizer.zero_grad()

                    try:
                        sample_batch_verb = next(trainloader_verb)
                        sample_batch_noun = next(trainloader_noun)
                    except:
                        break

                    out = self._model_output_fsl(sample_batch_verb, sample_batch_noun)
                    loss_verb = out["loss_verb"]
                    loss_noun = out["loss_noun"]
                    loss = 0.5* (loss_verb.item() + loss_noun.item())
                    losses.append(loss)

                    # Zero gradients, perform a backward pass, and update the weights.
                    
                    loss_verb.backward()
                    loss_noun.backward()

                    # Clipping Gradient
                    clip_grad_norm(self.model.parameters(), 20)

                    self.optimizer.step()

                    # For faster debugging
                    if self.debugging:
                        if idx == 20:
                            break
                    idx += 1

            elif self.cfg.algo == "IRM":
                self.model.train()
                train_loaders = [iter(x) for x in self.data_train]

                batch_idx = 0
                penalty_multiplier = (sum(epoch >= np.array([40, 60, 70]))) ** 1.1
                print("Using penalty multiplier = " + str(penalty_multiplier))

                num_env = 7
                data_sizes = [
                    len(ele.dataset) // self.cfg.train_batch_size
                    for ele in self.data_train
                ]
                idx = list(np.argsort(data_sizes)[::-1][:num_env])
                train_loaders = [train_loaders[p] for p in idx]
                data_sizes = [data_sizes[p] for p in idx]
                print(data_sizes)

                for env_idx in range(num_env):

                    errors = 0
                    penalties = 0
                    weight_norms = 0
                    total_loss = 0
                    self.optimizer.zero_grad()

                    for sample_batch in train_loaders[env_idx]:

                        out = self._model_output(sample_batch)
                        loss_erm = out["loss_erm"]
                        errors += loss_erm.mean().item()

                        # ERM Loss
                        loss = loss_erm.mean()

                        # Weight Normalization
                        if self.use_cuda:
                            weight_norm = torch.as_tensor(0.0).cuda()
                        else:
                            weight_norm = torch.as_tensor(0.0)

                        for w in self.model.parameters():
                            weight_norm += w.norm().pow(2)

                        #loss += 1e-5 * weight_norm
                        weight_norms += 1e-5 * weight_norm.item()

                        # IRM Loss
                        tmp_penalty = self.compute_irm_penalty(out["loss_verb"]) + self.compute_irm_penalty(out["loss_noun"])
                        loss += penalty_multiplier * tmp_penalty

                        penalties += penalty_multiplier * tmp_penalty.item()

                        total_loss += loss.item()
                        loss.backward()

                    # Clipping Gradient
                    clip_grad_norm(self.model.parameters(), 20)

                    losses.append((errors) / data_sizes[env_idx])

                    self.optimizer.step()

                    batch_idx += 1

                    # For faster debugging
                    if self.debugging:
                        if batch_idx == 20:
                            break


            if self.data_val is not None:
                if epoch % 10 == 0 or epoch == 22:

                    # Evaluation mode of model activated after each epoch
                    self.model.eval()
                    with torch.no_grad():
                        
                        losses_verb = []
                        losses_noun = []

                        correct_action = 0
                        correct_verb = 0
                        correct_noun = 0

                        # To count total number of instances
                        counter = 0

                        # Enumerate over batches
                        for idx, sample_batch in enumerate(self.data_val):
                            verb_anno = sample_batch["verb_id"]
                            noun_anno = sample_batch["noun_id"]

                            # if self.cfg.my_base_model:
                            out = self._model_output(sample_batch)
                            # else:
                            #     out = self._model_output_tsn(sample_batch)
                            output_verb = out["out_verb"]
                            output_noun = out["out_noun"]
                            loss_verb = out["loss_verb"]
                            loss_noun = out["loss_noun"]

                            # Accuracy Verb
                            if self.use_cuda:
                                pred_class_verb = torch.argmax(output_verb, dim=1).cpu()
                            else:
                                pred_class_verb = torch.argmax(output_verb, dim=1)

                            correct_verb += (
                                (verb_anno.view(-1) == pred_class_verb)
                                .float()
                                .sum()
                                .item()
                            )

                            # Accuracy Noun
                            if self.use_cuda:
                                pred_class_noun = torch.argmax(output_noun, dim=1).cpu()
                            else:
                                pred_class_noun = torch.argmax(output_noun, dim=1)

                            correct_noun += (
                                (noun_anno.view(-1) == pred_class_noun)
                                .float()
                                .sum()
                                .item()
                            )

                            # Accuracy Action
                            correct_action += (
                                (
                                    (verb_anno.view(-1) == pred_class_verb).int()
                                    & (noun_anno.view(-1) == pred_class_noun).int()
                                )
                                .float()
                                .sum()
                                .item()
                            )

                            losses_verb.append(loss_verb.mean().item())
                            losses_noun.append(loss_noun.mean().item())

                            counter += self.data_val.batch_size

                            # For faster debugging
                            if self.debugging:
                                if idx == 10:
                                    break

                        print(
                            "Epoch "
                            + str(epoch)
                            + ", val-seen loss = "
                            + str(0.5 * (np.mean(losses_verb) + np.mean(losses_noun)))
                        )

                        total_loss = 0.5 * (np.mean(losses_verb) + np.mean(losses_noun))

                        if not self.debugging:

                            # Save Model (Criteria: check every 10 epochs if the total val loss is less than the best recorded until now)
                            if epoch % 10 == 0:
                                if best_val_loss > total_loss:
                                    self.save_checkpoint(
                                        {
                                            "epoch": epoch,
                                            "model_state_dict": self.model.state_dict(),
                                            "optimizer_state_dict": self.optimizer.state_dict(),
                                            "loss": total_loss,
                                            "iter_num": iter_num,
                                            "summary_filename": self.summary_filename,
                                        }
                                    )
            else:
                if epoch % 10 == 0:
                    self.save_checkpoint(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "iter_num": iter_num,
                            "summary_filename": self.summary_filename,
                        },
                        intermediate=True,
                    )

            epoch += 1

        # Save the final fully trained model
        if not self.debugging:
            self.save_final_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "iter_num": iter_num,
                    "summary_filename": self.summary_filename,
                }
            )
            self.writer.close()

    def predict(self, filename):

        # Evaluation mode of model activated after each epoch
        self.model.eval()

        node = self.Node()
        import numpy as np

        with torch.no_grad():
            for i, sample_batch in enumerate(self.data_val):
                batch_size = sample_batch["data"].size(0)
                # compute output
                uids = sample_batch["uid"]
                output = self._model_test(sample_batch)
                # print(np.shape(uids))
                # print(np.shape(output["out_verb"]))
                for j in range(batch_size):
                    node.add(
                        uids[j], output["out_verb"][j, :], output["out_noun"][j, :]
                    )

                # if i == 3:
                #     break

        t = json.dumps(str(node))
        t = t.replace("'", '"')
        t = t[1:-1]
        with open(filename, "w") as json_file:
            json_file.write(t)
    
    def extract_features(self, sample_batch):
        return self._model_feature(sample_batch)
