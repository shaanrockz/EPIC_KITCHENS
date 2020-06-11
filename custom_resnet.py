from config import config
import torch
from torch import Tensor
from torch import nn
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock, model_urls, load_state_dict_from_url
from torch.nn.init import constant_, normal_
import torch.utils.checkpoint as checkpoint


class ResNetEPIC(ResNet):

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x


class EPICModel(nn.Module):
    def __init__(self, config, input_dim: int = 2048, drop_rate: float = 0.7):  # todo: try lower dropout
        super().__init__()
        self.cfg = config
        self.hidden_dim = input_dim
        self.num_verbs = config.num_class_verb
        self.num_nouns = config.num_class_noun
        self.drop_rate = drop_rate
        self.base = resnet50_without_fc()
        self.num_crop = 1
        self.dropout = nn.Dropout(p=self.drop_rate)

        self.verb_layer = nn.Linear(
            in_features=self.hidden_dim, out_features=self.num_verbs)
        self.noun_layer = nn.Linear(
            in_features=self.hidden_dim, out_features=self.num_nouns)
        self._initialise_layer(self.verb_layer)
        self._initialise_layer(self.noun_layer)

        if not self.cfg.feature_extraction:
            self.dummy_tensor = torch.ones(
                1, dtype=torch.float32, requires_grad=True)
            self.module_wrapper = ModuleWrapperIgnores2ndArg(self.base)

    def _initialise_layer(self, layer, mean=0, std=0.001):
        normal_(layer.weight, mean, std)
        constant_(layer.bias, mean)

    def forward_consensus_verb(self, x):
        verb_logits = self.verb_layer.forward(
            x)  # [(B x num_segments x crops) x num_verbs]
        #verb_logits = self.tanh(verb_logits)
        verb_logits_reshaped = verb_logits.view(
            (-1, self.cfg.n_segments*self.num_crop, self.num_verbs))
        verb_logits_consensus = torch.mean(verb_logits_reshaped, dim=1)
        return verb_logits_consensus

    def forward_consensus_noun(self, x):
        noun_logits = self.noun_layer.forward(
            x)  # [(B x num_segments x crops) x num_nouns]
        #noun_logits = self.tanh(noun_logits)
        noun_logits_reshaped = noun_logits.view(
            (-1, self.cfg.n_segments*self.num_crop, self.num_nouns))
        noun_logits_consensus = torch.mean(noun_logits_reshaped, dim=1)
        return noun_logits_consensus

    def forward(self, data: Tensor):
        """
        input: [B x 3 x 224 x 224]
        input: [B x num_segments x 3 x 224 x 224]
        """
        input_reshaped = data.view(-1, 3, 224, 224)

        if not self.cfg.feature_extraction:
            x = checkpoint.checkpoint(
                self.module_wrapper, input_reshaped, self.dummy_tensor)
        else:
            x = self.base(input_reshaped)

        x = self.dropout(x)

        if not self.cfg.feature_extraction:
            verb_logits_consensus = checkpoint.checkpoint(
                self.forward_consensus_verb, x)
            noun_logits_consensus = checkpoint.checkpoint(
                self.forward_consensus_noun, x)
        else:
            verb_logits_consensus = self.forward_consensus_verb(x)
            noun_logits_consensus = self.forward_consensus_noun(x)

        return {"out_shared": x, "out_verb": verb_logits_consensus, "out_noun": noun_logits_consensus}

    def train(self, mode):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        if mode is None:
            self.num_crop = 1
            print("training mode ON Normal")
            super(EPICModel, self).train(True)
            self.base.train(True)
            count = 0
            if self.cfg._enable_pbn:
                print("Freezing BatchNorm2D except the first one.")
                for m in self.base.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
        else:
            if mode:
                self.num_crop = 1
                print("training mode ON Normal")
                super(EPICModel, self).train(True)
                self.base.train(True)
                count = 0
                if self.cfg._enable_pbn:
                    print("Freezing BatchNorm2D except the first one.")
                    for m in self.base.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            count += 1
                            if count >= 2:
                                m.eval()
                                # shutdown update in frozen mode
                                m.weight.requires_grad = False
                                m.bias.requires_grad = False
                

    def eval(self):
        print("training mode OFF")
        self.base.eval()
        super(EPICModel, self).eval()
        self.num_crop = 10

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self.cfg._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]


def my_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetEPIC(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_without_fc(pretrained=True, progress=True, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return my_resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

    # return my_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


