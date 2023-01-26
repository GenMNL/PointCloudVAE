import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json


"""collateの作成
"""
class CollateUpSampling():
    def __init__(self, device):
        self.device = device

    def __call__(self, batch_list):
        # get batch size
        batch_size = len(batch_list)

        # * in *batch_list is transpose of batch_list
        # There are as many tensors as there are batchsize in batch_list
        # comp_batch and partial_batch are tuple which include many tensors
        # input_batch, truth_batch, a, b, c, d = list(zip(*batch_list))
        # input_batch, truth_batch, a, b = list(zip(*batch_list))
        input_batch = list(zip(*batch_list))
        # transform tuple of complete point cloud to tensor
        input_batch = list(input_batch)

        # get max num points in each batch
        max_num_points = 0
        for j in range(batch_size):
            n = len(input_batch[j])
            if max_num_points < n:
                max_num_points = n

        # up sampling
        unique_mask_list = [0]*batch_size
        for i in range(batch_size):
            n = len(input_batch[i]) # num of each tensor in bach
            idx = np.random.permutation(n)

            if n < max_num_points:
                unique_idx = np.random.randint(0, n, size=(max_num_points - n)) # unique indecies
                idx = np.concatenate([idx, unique_idx])
            input_batch[i] = input_batch[i][idx, :]

            # make unique mask for removing unique points in loss and differential term
            unique_mask = torch.zeros_like(input_batch[i], dtype=torch.bool, device=self.device) 
            unique_mask[n:, :] = True
            unique_mask_list[i] = unique_mask

        # torch.stack concatenate each tensors in the direction of the specified dim(dim=0)
        input_batch = torch.stack(input_batch, dim=0).to(self.device).permute(0, 2, 1)
        unique_mask_batch = torch.stack(unique_mask_list, dim=0).to(self.device).permute(0, 2, 1)

        return input_batch, unique_mask_batch

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
