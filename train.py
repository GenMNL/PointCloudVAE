import torch
from torch.utils.data import DataLoader
import numpy as np
import parser
import datetime
import os
import json
from tqdm import tqdm
from data import *
from model import PointVAE
from options import make_parser

def train_one_epoch(model, dataloader, optim):
    model.train()

    count = 0
    sum_train_loss = 0.0
    for i, data in enumerate(tqdm(dataloader, desc="train"), leave=False):
        point_cloud = data[0]

        _, prediction = model(point_cloud)

        sum_train_loss += train_loss
        count += 1

    sum_train_loss /= count
    return sum_train_loss

def val_one_epoch(model, dataloader):
    model.eval()

    count = 0
    sum_val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="validation"), leave=False):
            point_cloud = data[0]

            _, prediction = model(point_cloud)

if __name__ == "__main__":

    # prepare options
    args = make_parser()
    args = parser.parse_args()

    # make path of save params
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    dt_now = datetime.now()
    save_date = str(dt_now.month) + str(dt_now.day) + "-" + str(dt_now.hour) + "-" +str(dt_now.minute)
    if not os.path.exists(os.path.join(args.save_dir, str(dt_now.year))):
        os.mkdir(os.path.join(args.save_dir, str(dt_now.year)))
    save_dir = os.path.join(args.save_dir, str(dt_now.year), save_date)
    save_normal = os.path.join(save_dir, "normal_parameters.tar")
    save_best = os.path.join(save_dir, "best_parameters.tar")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        with open(os.path.join(save_dir, "conditions.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)


    # make dataloader
    if args.subset == "all":
        subset_id = "all"
    else:
        subset_id_path = os.path.join(args.dataset_dir, "synsetoffset2category.txt")
        with open(subset_id_path, "r") as subset_id_list:
            subset_dict = {}
            for i in subset_id_list:
                name, subset_id = i.split()
                subset_dict[name] = subset_id 

            if args.subset in subset_dict:
                subset_id = subset_dict[args.subset]

    # training data
    train_dataset = MakeDataset(path=args.dataset_dir, eval="train",
                                subset=subset_id, device=args.device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch,
                                  shuffle=True, drop_last=True,
                                  collate_fn=CollateTrain(args.device)) # DataLoader is iterable object.
    val_dataset = MakeDataset(path=args.dataset_dir, eval="val",
                                subset=subset_id, device=args.device)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch,
                                  shuffle=True, drop_last=True,
                                  collate_fn=CollateTrain(args.device)) # DataLoader is iterable object.

    # prepare model
    Model = PointVAE(in_dim=3, num_out_points=args.num_out_points, z_dim=args.z_dim)
    optim = torch.optim.Adam(params=Model.parameters(), lr=args.lr)


    torch.autograd.set_detect_anomaly(True)
    # main loop
    best_loss = np.inf
    for epoch in tqdm(range(1, args.epochs), desc="main loop"):
        train_loss = train_one_epoch(Model, train_dataloader, optim)
        val_loss = val_one_epoch(Model, val_dataloader)


        torch.save({
            "epoch": epoch,
            "model_state_dict": Model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "loss": train_loss
        }, save_normal)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": Model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": best_loss
            }, save_best)
