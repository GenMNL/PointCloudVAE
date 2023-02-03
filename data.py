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

        input_batch, subset_name_bach = list(zip(*batch_list))
        input_batch = list(input_batch)
        subset_name_list = list(subset_name_bach)

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
        subset_name_array = np.array(list(subset_name_list))

        return input_batch, unique_mask_batch, subset_name_array

class DataNormalization():
    def __init__(self):
        pass

    def __call__(self, tensor):
        """0-1 normalization
        Args:
            beam_tensor (tensor): (N, C)
        Returns:
            tensor: (N, C)
        """

        max_values = torch.max(tensor, dim=0, keepdim=True)[0]
        min_values = torch.min(tensor, dim=0, keepdim=True)[0]

        normalized_tensor = (tensor - min_values)/(max_values - min_values)
        return normalized_tensor, max_values, min_values
# ----------------------------------------------------------------------------------------

class MakeDataset(Dataset):
    def __init__(self, points_path_list, subset_name_list, device, transform=DataNormalization):
        super().__init__()
        self.points_path_list = points_path_list
        self.subset_name_list = subset_name_list
        self.device = device
        self.transform = transform() # min-max normalization

    def __len__(self):
        return len(self.points_path_list)

    def __getitem__(self, index):
        points_path_list = self.points_path_list[index]
        points = np.loadtxt(points_path_list).astype(np.float32)
        points = torch.tensor(points, device=self.device)

        subset_name_list = self.subset_name_list[index]
        return points, subset_name_list

def get_item_from_json(dataset_path, eval, subset_id):
    json_path = os.path.join(dataset_path, "train_test_split")
    # read json file
    with open(f"{json_path}/shuffled_{eval}_file_list.json", "r") as f:
        data_list = json.load(f)

    # get the id and index of object which wants to train(or test)
    points_path_list = []
    subset_name_list = []
    for i in range(len(data_list)):
        full_path = data_list[i].split("/")
        if subset_id == "all":
            points_file = os.path.join(dataset_path, str(full_path[1]), "points", str(full_path[2])+".pts")
            points_path_list.append(points_file)
        else:
            if str(full_path[1]) == subset_id:
                points_file = os.path.join(dataset_path, str(full_path[1]), "points", str(full_path[2])+".pts")
                points_path_list.append(points_file)
        
        subset_name = search_subset(dataset_path, str(full_path[1]))
        subset_name_list.append(subset_name)

    return points_path_list, subset_name_list

def search_subset(dataset_path, subset_id):

    subset_id_path = os.path.join(dataset_path, "synsetoffset2category.txt")
    with open(subset_id_path, "r") as subset_id_list:
        subset_dict = {}
        for i in subset_id_list:
            name, id = i.split()
            subset_dict[id] = name

    return subset_dict[subset_id]
