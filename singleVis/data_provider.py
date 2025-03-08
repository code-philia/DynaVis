import os
import torch
import numpy as np
import re

class DataProvider:
    def __init__(self, content_path, epoch_start, epoch_end, epoch_period, split, selected_groups=None):
        self.content_path = content_path
        self.s = epoch_start
        self.e = epoch_end
        self.p = epoch_period
        self.split = split
        self.selected_groups = selected_groups or []
        self.train_epochs = self._list_train_epochs()

    def _list_train_epochs(self):
        train_files = os.listdir(self.content_path)
        epochs = dict()
        pattern = re.compile(r"Epoch_(\d+)_train_data_(\d+).npy")
        for file in train_files:
            match = pattern.match(file)
            if match:
                epoch = int(match.group(1))
                group = int(match.group(2))
                if group not in self.selected_groups:
                    continue
                if epoch not in epochs:
                    epochs[epoch] = []
                epochs[epoch].append(os.path.join(self.content_path, file))
        return epochs

    def train_representation(self, epoch):
        try:
            self.train_epochs = self._list_train_epochs()
            train_files = self.train_epochs.get(epoch, [])
            if not train_files:
                return None
            train_data = []
            for file in train_files:
                data = np.load(file)
                train_data.append(data)
            return np.vstack(train_data)
        except:
            return None

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