import os
import torch
import numpy as np
import re

class DataProvider:
    def __init__(self, content_path, epoch_start, epoch_end, epoch_period, select_idxs=None):
        self.content_path = content_path
        self.s = epoch_start
        self.e = epoch_end
        self.p = epoch_period
        self.select_idxs = select_idxs
        self.train_epochs = self._list_train_epochs()

    def _list_train_epochs(self):
        train_epochs = {}
        for dir_name in os.listdir(self.content_path):
            dir_path = os.path.join(self.content_path, dir_name)
            if os.path.isdir(dir_path):
                match = re.match(r"Epoch_(\d+)", dir_name)
                if match:
                    epoch = int(match.group(1))
                    if self.s <= epoch <= self.e and (epoch - self.s) % self.p == 0:
                        data_file = os.path.join(dir_path, "train_data.npy")
                        if os.path.isfile(data_file):
                            train_epochs[epoch] = data_file
        return train_epochs

    def train_representation(self, epoch, select_sample=None):
        try:
            if select_sample is not None:
                select_idxs = select_sample
            else:
                select_idxs = self.select_idxs

            dir_name = f"Epoch_{epoch}"
            dir_path = os.path.join(self.content_path, dir_name)
            data_path = os.path.join(dir_path, "train_data.npy")

            if not os.path.isfile(data_path):
                print(f"Epoch {epoch} not found in train_epochs.")
                return None

            data = np.load(data_path)

            if select_idxs is not None:
                data = data[select_idxs]

            return data
        except Exception as e:
            print(f"Error loading data for Epoch {epoch}: {e}")
            return None

    def get_available_epochs(self):
        return sorted(list(self.train_epochs.keys()))
    
import os
import numpy as np
import re

class NewDataProvider:
    def __init__(self, content_path, epoch_start, epoch_end, epoch_period, split, max_samples=None):
        self.content_path = content_path
        self.s = epoch_start
        self.e = epoch_end
        self.p = epoch_period
        self.split = split
        self.train_epochs = self._list_train_epochs()
        self.max_samples = max_samples  # 新增参数：控制每次读取的最大样本数

    def _list_train_epochs(self):
        train_epochs = {}
        for dir_name in os.listdir(self.content_path):
            dir_path = os.path.join(self.content_path, dir_name)
            if os.path.isdir(dir_path):
                match = re.match(r"Epoch_(\d+)", dir_name)
                if match:
                    epoch = int(match.group(1))
                    code_file = os.path.join(dir_path, "train_code_cls_tokens.npy")
                    nl_file = os.path.join(dir_path, "train_nl_cls_tokens.npy")
                    if os.path.isfile(code_file) and os.path.isfile(nl_file):
                        train_epochs[epoch] = {
                            "code_path": code_file,
                            "nl_path": nl_file
                        }
        return train_epochs

    def train_representation(self, epoch):
        try:
            epoch_info = self.train_epochs.get(epoch)
            if not epoch_info:
                print(f"Epoch {epoch} not found in train_epochs.")
                return None
            
            code_data = np.load(epoch_info["code_path"])
            nl_data = np.load(epoch_info["nl_path"])
            
            if self.max_samples is None or self.max_samples >= len(code_data):
                combined_data = np.vstack((code_data, nl_data))
                return combined_data
            else:
                code_data = code_data[:self.max_samples]
                nl_data = nl_data[:self.max_samples]
                combined_data = np.vstack((code_data, nl_data))
                return combined_data
        except Exception as e:
            print(f"Error loading data for Epoch {epoch}: {e}")
            return None