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


