import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json


"""collateの作成
"""





class MakeDataset(Dataset):
    def __init__(self, path, eval, subset, device):
        super().__init__()
        self.path = path
        self.eval = eval
        self.subset = subset
        self.device = device

    def __len__(self):
        points_path_list = self.get_item_from_json()

        return len(points_path_list)

    def __getitem__(self, index):
        points_path_list = self.get_item_from_json()

        points_path_list = points_path_list[index]
        points = np.loadtxt(points_path_list).astype(np.float32)

        return points

    def get_item_from_json(self):
        json_path = os.path.join(self.dataset_path, "train_test_split")
        # read json file
        read_json = open(f"{json_path}/shuffled_{self.eval}_file_list.json", "r")
        self.data_list = json.load(read_json)

        # get the id and index of object which wants to train(or test)
        points_path_list = []
        for i in range(len(self.data_list)):
            full_path = self.data_list[i].split("/")
            if self.subset == "all":
                points_file = os.path.join(self.dataset_path, str(full_path[1]), "points", str(full_path[2])+".pts")
                points_path_list.append(points_file)
            else:
                if str(full_path[1]) == self.subset:
                    points_file = os.path.join(self.dataset_path, str(full_path[1]), "points", str(full_path[2])+".pts")
                    points_path_list.append(points_file)

        return points_path_list
