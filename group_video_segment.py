import pandas as pd
import os


class GroupSegment:
    def __init__(self, config, setname, test_mode=None, participant_id=None):
        self.cfg = config
        self.n_segments = self.cfg.n_segments
        self.setname = setname

        # Reading Annotation file
        if not self.setname == "test":
            self.df = pd.read_csv(self.cfg.anno_path)
        elif test_mode == "seen":
            self.df = pd.read_csv(self.cfg.anno_path_seen)
        elif test_mode == "unseen":
            self.df = pd.read_csv(self.cfg.anno_path_unseen)

        # Accumulating data index relevent for setname mode
        self.items = []
        self.verb_labels = []
        self.noun_labels = []
        for idx, item in self.df.iterrows():
            if self.setname == "val-unseen":
                if int(item.participant_id[1:]) >= 26:
                    self.items.append(idx)
                    self.verb_labels.append(int(item.verb_class))
                    self.noun_labels.append(int(item.noun_class))
            elif self.setname == "train-unseen":
                if int(item.participant_id[1:]) < 26:
                    if participant_id:
                        if item.participant_id == participant_id:
                            self.items.append(idx)
                            self.verb_labels.append(int(item.verb_class))
                            self.noun_labels.append(int(item.noun_class))
                    else:
                        self.items.append(idx)
                        self.verb_labels.append(int(item.verb_class))
                        self.noun_labels.append(int(item.noun_class))
            elif self.setname == "val-seen":
                if item["video_id"] in self.cfg.val_seen_vids:
                    self.items.append(idx)
                    self.verb_labels.append(int(item.verb_class))
                    self.noun_labels.append(int(item.noun_class))
            elif self.setname == "train-seen":
                if item["video_id"] not in self.cfg.val_seen_vids:
                    if participant_id:
                        if item.participant_id == participant_id:
                            self.items.append(idx)
                            self.verb_labels.append(int(item.verb_class))
                            self.noun_labels.append(int(item.noun_class))
                    else:
                        self.items.append(idx)
                        self.verb_labels.append(int(item.verb_class))
                        self.noun_labels.append(int(item.noun_class))
            elif self.setname == "whole-train":
                if participant_id:
                    if item.participant_id == participant_id:
                        self.items.append(idx)
                        self.verb_labels.append(int(item.verb_class))
                        self.noun_labels.append(int(item.noun_class))
                else:
                    self.items.append(idx)
                    self.verb_labels.append(int(item.verb_class))
                    self.noun_labels.append(int(item.noun_class))
            elif self.setname == "test":
                self.items.append(idx)
                self.verb_labels.append(int(item.verb_class))
                self.noun_labels.append(int(item.noun_class))
        
        if len(self.items) == 0:
            self.haveData = False
        else:
            self.haveData = True

    def _process_snapshot_(self, idx):
        # Getting global Index of dataframe
        idx = self.items[idx]

        self.dataFrame = self.df.iloc[idx]

        if self.setname == "test":
            self.uid = self.dataFrame["uid"]
        else:
            # Extracting Verb and Noun annotation
            self.ann_verb = self.dataFrame["verb_class"]
            self.ann_noun = self.dataFrame["noun_class"]

        # Extracting Frame #
        self.low_idx, self.high_idx = (
            int(self.dataFrame["start_frame"]),
            int(self.dataFrame["stop_frame"]),
        )

        self.segment_len = int((self.high_idx - self.low_idx) / self.n_segments)

        # Image folder path for given participant id and video id
        self.img_path_global = (
            self.cfg.data_path
            + self.dataFrame["participant_id"]
            + "/"
            + self.dataFrame["video_id"]
            + "/"
        )

    def _get_image_path_(self, segment_indx):
        """ Given a segment index, lower_index_for_instance and segment_length we 
        evaluate index of the frame to consider. From each segments center frame 
        is considered for training and validation"""
        path = os.path.join(
            self.img_path_global,
            self.cfg.image_tmpl.format(
                self.low_idx
                + (segment_indx * self.segment_len)
                + int(self.segment_len / 2)
            ),
        )
        return path
