import yaml


class config:
    def __init__(self, external_path=None, debugging=False, additionalText='sample'):

        if external_path:
            stream = open(external_path, "r")
            docs = yaml.load_all(stream)
            for doc in docs:
                for k, v in doc.items():
                    #if k == "train":
                    for k1, v1 in v.items():
                        cmd = "self." + k1 + "=" + repr(v1)
                        print(cmd)
                        exec(cmd)
                        # self.__setattr__(k1, repr(v1))
            stream.close()
            if hasattr(self, 'train_mode'):
                assert self.train_mode in ["train-seen", "train-unseen", "val-seen", "val-unseen", "whole-train", "test"]
            if hasattr(self, 'val_mode'):
                assert self.val_mode in ["train-seen", "train-unseen", "val-seen", "val-unseen", "whole-train", "test"]
        else:
            self.manualSeed = 786
            self.epoch = 80

            # Dataset Config
            self.anno_path = "/media/data/salam/epic-kitchens/annotations/EPIC_train_action_labels.csv"
            self.data_path = "/media/data/salam/epic-kitchens/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train/"
            self.train_mode = "train-seen"
            self.val_mode = "val-seen"
            
            if hasattr(self, 'train_mode'):
                assert self.train_mode in ["train-seen", "train-unseen", "val-seen", "val-unseen", "whole-train", "test"]
            if hasattr(self, 'val_mode'):
                assert self.val_mode in ["train-seen", "train-unseen", "val-seen", "val-unseen", "whole-train", "test"]
            self.val_seen_vids = [
                "P01_18",
                "P02_09",
                "P03_27",
                "P04_04",
                "P07_07",
                "P08_20",
                "P15_01",
                "P22_08",
                "P24_03",
                "P25_01",
                "P26_05",
                "P28_02",
                "P30_04",
                "P31_04",
            ]

            # Model Config
            self.n_segments = 8
            self.algo = "MTGA"

        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 5e-4

        self.num_class_verb = 125
        self.num_class_noun = 352
        
        self.image_tmpl = "frame_{:010d}.jpg"

        self._enable_pbn = True

        if debugging:
            self.epoch = 1
            self.train_batch_size = 2
            self.val_batch_size = 2
            self.num_worker_train = 0
            self.num_worker_val = 0
            self.load_model = False
            self.additional_info = 'Debug_mode'
        else:
            if self.algo == "IRM":
                self.train_batch_size = 16*4
                self.val_batch_size = 8
                self.num_worker_train = 1
                self.num_worker_val = 4
            elif self.algo == "ERM":
                self.train_batch_size = 16*2
                self.val_batch_size = 8*2
                self.num_worker_train = 16
                self.num_worker_val = 8
            elif self.algo == "MTGA":
                self.train_batch_size = 16*2
                self.val_batch_size = 8*2
                self.num_worker_train = 16
                self.num_worker_val = 8
            elif self.algo == "FSL":
                # For few shot learning
                self.shot = 1
                self.way = 3
                self.query = 4
                self.temperature = 1
                self.train_batch_size = self.way * (self.shot + self.query) 
                self.val_batch_size = 8
                self.num_worker_train = 8
                self.num_worker_val = 4
            self.load_model = False
            self.feature_extraction = False
            self.additional_info = additionalText

        # Evaluation Config

        self.checkpoint_filename = "checkpoint_" + \
            "_train_config= "+str(self.additional_info)+".pth"
        self.checkpoint_filename_final = "checkpoint_model_final.pth"
