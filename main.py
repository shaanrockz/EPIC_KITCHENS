import os
import torch
import random
from dataset import EPIC
from config import config
from experiment import Experiment
from torch.utils.data import DataLoader
from custom_resnet import EPICModel
from samplers import CategoriesSampler
import sys
from optparse import OptionParser
import transforms
import torchvision
import pandas as pd
import numpy as np

# Parser options
parser = OptionParser()
parser.add_option("--gpu", type=int, help="gpu id", default=1)
parser.add_option("--config", type=str, help="configuration")


def main(argv):
    # Read arguments passed
    (opts, args) = parser.parse_args(argv)

    # Reading config
    cfg = config(opts.config, debugging=False, additionalText="training_MTGA_unseen_resnet18")

    # Use CUDA
    # os.environ['CUDA_VISIBLE_DEVICES'] = 1
    use_cuda = torch.cuda.is_available()

    # If the manual seed is not yet choosen
    if cfg.manualSeed == None:
        cfg.manualSeed = 1

    # Set seed for reproducibility for CPU and GPU randomizaton process
    random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)

    if use_cuda:
        torch.cuda.manual_seed_all(cfg.manualSeed)

    dataloader_train = None
    if hasattr(cfg, "train_mode"):

        # Preprocessing (transformation) instantiation for training groupwise
        transformation_train = torchvision.transforms.Compose(
            [
                transforms.GroupMultiScaleCrop(224, [1, 0.875, 0.75, 0.66]),
                transforms.GroupRandomHorizontalFlip(is_flow=False),
                transforms.Stack(),  # concatenation of images
                transforms.ToTorchFormatTensor(),  # to torch
                transforms.GroupNormalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalization
            ]
        )

        if cfg.algo == "ERM" or cfg.algo == "MTGA":
            # Loading training Dataset with N segment for TSN
            EPICdata_train = EPIC(
                mode=cfg.train_mode, cfg=cfg, transforms=transformation_train,
            )

            # Creating validation dataloader
            # batch size = 16, num_workers = 8 are best fit for 12 Gb GPU and >= 16 Gb RAM
            dataloader_train = DataLoader(
                EPICdata_train,
                batch_size=cfg.train_batch_size,
                shuffle=True,
                num_workers=cfg.num_worker_train,
                pin_memory=True,
            )
        elif cfg.algo == "IRM":
            df = pd.read_csv(cfg.anno_path)
            p_ids = list(set(df["participant_id"].tolist()))
            
            dataloader_train = []
            for p_id in p_ids:
                tmp_dataset = EPIC(
                    mode=cfg.train_mode,
                    cfg=cfg,
                    transforms=transformation_train,
                    participant_id=p_id,
                )

                if tmp_dataset.haveData:
                    dataloader_train.append(
                        DataLoader(
                            tmp_dataset,
                            batch_size=cfg.train_batch_size,
                            shuffle=True,
                            num_workers=cfg.num_worker_train,
                            pin_memory=True,
                        )
                    )
        elif cfg.algo == "FSL":
            dataloader_train = {}
            # Loading training Dataset with N segment for TSN
            EPICdata_train_verb = EPIC(
                mode=cfg.train_mode, cfg=cfg, transforms=transformation_train
            )
            sampler = CategoriesSampler(EPICdata_train_verb.verb_label, 200, cfg.way, cfg.shot + cfg.query)
            dataloader_train["verb"] = DataLoader(
                        dataset = EPICdata_train_verb,
                        batch_sampler=sampler,
                        num_workers=cfg.num_worker_train,
                        pin_memory=True,
                    )
            
            EPICdata_train_noun = EPIC(
                mode=cfg.train_mode, cfg=cfg, transforms=transformation_train
            )
            sampler = CategoriesSampler(EPICdata_train_noun.noun_label, 200, cfg.way, cfg.shot + cfg.query)
            dataloader_train["noun"] = DataLoader(
                        dataset = EPICdata_train_noun,
                        batch_sampler=sampler,
                        num_workers=cfg.num_worker_train,
                        pin_memory=True,
                    )



    dataloader_val = None
    if hasattr(cfg, "val_mode") and hasattr(cfg, "train_mode"):
        # Preprocessing (transformation) instantiation for validation groupwise
        transformation_val = torchvision.transforms.Compose(
            [
                transforms.GroupOverSample(
                    224, 256
                ),  # group sampling from images using multiple crops
                transforms.Stack(),  # concatenation of images
                transforms.ToTorchFormatTensor(),  # to torch
                transforms.GroupNormalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalization
            ]
        )
        
        # Loading validation Dataset with N segment for TSN
        EPICdata_val = EPIC(mode=cfg.val_mode, cfg=cfg, transforms=transformation_val,)

        # Creating validation dataloader
        dataloader_val = DataLoader(
            EPICdata_val,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=cfg.num_worker_val,
            pin_memory=True,
        )

    # Loading Models (Resnet50)
    model = EPICModel(config=cfg)

    if not cfg.feature_extraction:
        if hasattr(cfg, "train_mode"):
            policies = model.get_optim_policies()

            # for group in policies:
            #     print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            #         group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

            # Optimizer
            # initial lr = 0.01
            # momentum = 0.9
            # weight_decay = 5e-4
            optimizer = torch.optim.SGD(
                policies, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
            )

            # Loss function (CrossEntropy)
            if cfg.algo == "IRM":
                criterion = torch.nn.CrossEntropyLoss(reduction="none")
            elif cfg.algo == "ERM" or cfg.algo == "MTGA":
                criterion = torch.nn.CrossEntropyLoss()
            elif cfg.algo == "FSL":
                criterion = torch.nn.CrossEntropyLoss()

            # If multiple GPUs are available (and bridged)
            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     model = torch.nn.DataParallel(model)

            # Convert model and loss function to GPU if available for faster computation
            if use_cuda:
                model = model.cuda()
                criterion = criterion.cuda()

            # Loading Trainer
            experiment = Experiment(
                cfg=cfg,
                model=model,
                loss=criterion,
                optimizer=optimizer,
                use_cuda=use_cuda,
                data_train=dataloader_train,
                data_val=dataloader_val,
                debugging=False,
            )

            # Train the model
            experiment.train()

        else:

            # Load Model Checkpoint
            checkpoint = torch.load(cfg.checkpoint_filename_final)
            model.load_state_dict(checkpoint["model_state_dict"])

            if use_cuda:
                model = model.cuda()

            transformation = torchvision.transforms.Compose(
                [
                    transforms.GroupOverSample(
                        224, 256
                    ),  # group sampling from images using multiple crops
                    transforms.Stack(),  # concatenation of images
                    transforms.ToTorchFormatTensor(),  # to torch
                    transforms.GroupNormalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # Normalization
                ]
            )

            # Loading Predictor
            experiment = Experiment(
                cfg=cfg, model=model, use_cuda=use_cuda, debugging=False
            )

            filenames = ["seen.json", "unseen.json"]
            for filename in filenames:
                EPICdata = EPIC(
                    mode=cfg.val_mode,
                    cfg=cfg,
                    transforms=transformation,
                    test_mode=filename[:-5],
                )

                data_loader = torch.utils.data.DataLoader(
                    EPICdata, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
                )
                experiment.data_val = data_loader
                experiment.predict(filename)
    else:
        # Load Model Checkpoint
        checkpoint = torch.load(cfg.checkpoint_filename_final)
        model.load_state_dict(checkpoint["model_state_dict"])

        if use_cuda:
            model = model.cuda()

        model.eval()

        transformation = torchvision.transforms.Compose(
            [
                transforms.GroupOverSample(
                    224, 256
                ),  # group sampling from images using multiple crops
                transforms.Stack(),  # concatenation of images
                transforms.ToTorchFormatTensor(),  # to torch
                transforms.GroupNormalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalization
            ]
        )

        # Loading Predictor
        experiment = Experiment(
            cfg=cfg, model=model, use_cuda=use_cuda, debugging=False
        )

        with torch.no_grad():
            modes = ["train-unseen", "val-unseen"]
            for mode in modes:
                data = np.empty((1, 2050))
                EPICdata = EPIC(
                    mode=mode,
                    cfg=cfg,
                    transforms=transformation,
                )

                data_loader = torch.utils.data.DataLoader(
                    EPICdata, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
                )

                for i, sample_batch in enumerate(data_loader):
                    output = experiment.extract_features(sample_batch)
                    verb_ann = sample_batch["verb_id"].data.item()
                    noun_ann = sample_batch["noun_id"].data.item()
                    out = np.append(np.mean(output, 0), verb_ann)
                    out = np.append(out, noun_ann)
                    data = np.concatenate((data, np.expand_dims(out, 0)), 0)
                    # print(np.shape(data))
                    # print(data)
                    # break
                np.save(mode, data)


if __name__ == "__main__":
    main(sys.argv)
