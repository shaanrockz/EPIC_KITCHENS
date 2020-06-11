from torch import Tensor
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from typing import Dict
from group_video_segment import GroupSegment


class EPIC(Dataset):
    def __init__(self, mode, cfg, transforms=None, test_mode=None, participant_id=None):
        super().__init__()
        self.cfg = cfg
        self.n_segments = self.cfg.n_segments
        assert mode in ["train-seen", "train-unseen", "val-seen", "val-unseen", "whole-train", "test"]
        self.setname = mode
        self.transforms = transforms
        self.test_mode = test_mode
        # Define Group Segment Extractor
        self.group_segment = GroupSegment(self.cfg, self.setname, self.test_mode, participant_id)
        self.haveData = self.group_segment.haveData
        self.verb_label = self.group_segment.verb_labels
        self.noun_label = self.group_segment.noun_labels

    def __len__(self) -> int:
        return len(self.group_segment.items)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        self.group_segment._process_snapshot_(idx)

        # Looping over # of segments
        data = []
        for seg_indx in range(self.n_segments):
            tmp = Image.open(self.group_segment._get_image_path_(seg_indx)).convert(
                "RGB"
            )
            data.append(tmp)

        if self.transforms:
            data = self.transforms(data)

        # PyTorch automatically converts data to Torch Tensors
        if self.setname == "test":
            return {
                "data": data,
                "uid": self.group_segment.uid
            }
        else:
            return {
                "data": data,
                "verb_id": torch.Tensor(
                    np.expand_dims(self.group_segment.ann_verb, 1)
                ).type(torch.LongTensor),
                "noun_id": torch.Tensor(
                    np.expand_dims(self.group_segment.ann_noun, 1)
                ).type(torch.LongTensor),
            }

import torchvision
import transforms
from config import config
from torch.utils.data import DataLoader
if __name__ == "__main__":

    # Reading config
    cfg = config()

    # Preprocessing (transformation) instantiation for training groupwise
    transformation = torchvision.transforms.Compose(
        [
            transforms.GroupScale(256),  # scale images
            transforms.GroupCenterCrop(224),  # center crop images
            transforms.Stack(),  # concatenation of images
            transforms.ToTorchFormatTensor(),  # to torch
            transforms.GroupNormalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalization
        ]
    )

    EPICdata = EPIC(
        mode=cfg.train_mode,
        cfg=cfg,
        transforms=transformation,
    )

    # Creating validation dataloader
    # batch size = 16, num_workers = 8 are best fit for 12 Gb GPU and >= 16 Gb RAM
    dataloader = DataLoader(
        EPICdata, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_worker_train, pin_memory=True
    )

    for idx, sample_batch in enumerate(dataloader):
        images = sample_batch['data']
        print(images.shape)
        images = images.view(-1, 3, 224, 224)
        print(images.shape)
        print("In debugging mode")
