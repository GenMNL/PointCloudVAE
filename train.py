import torch
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
from pytorch3d.loss import chamfer_distance
import parser
import datetime
import os
import json
from tqdm import tqdm
from data import *
from model import PointVAE
from options import make_parser

# --------------------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optim):
    model.train()

    sum_cd_loss = 0.0
    # sum_mse_loss = 0.0
    sum_kl_loss = 0.0
    sum_train_loss = 0.0
    for i, data in enumerate(tqdm(dataloader, desc="train", leave=False)):
        # load data
        original_point_cloud = data[0]

        # get prediction
        mu, log_var, _, prediction = model(original_point_cloud)
        # _, prediction = model(original_point_cloud)

        # reshape for cal cd_loss
        original_point_cloud = original_point_cloud.permute(0, 2, 1)

        # cal loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
        # mse_loss = torch.sum((prediction - original_point_cloud)**2)
        # train_loss = kl_loss + mse_loss
        cd_loss, _ = chamfer_distance(prediction, original_point_cloud)
        train_loss = kl_loss + 5e7*cd_loss

        # optimization
        optim.zero_grad()
        train_loss.backward()
        optim.step()

        # sum train loss
        # sum_mse_loss += mse_loss
        sum_cd_loss += cd_loss
        sum_kl_loss += kl_loss
        sum_train_loss += train_loss

    # sum_mse_loss /= len(dataloader)
    sum_cd_loss /= len(dataloader)
    sum_kl_loss /= len(dataloader)
    sum_train_loss /= len(dataloader)
    return sum_cd_loss, sum_kl_loss, sum_train_loss
    # return sum_train_loss

def val_one_epoch(model, dataloader):
    model.eval()

    sum_val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="validation", leave=False)):
            # load data
            original_point_cloud = data[0]

            # get prediction
            mu, log_var, _, prediction = model(original_point_cloud)
            # _, prediction = model(original_point_cloud)

            # reshape for cal cd_loss
            original_point_cloud = original_point_cloud.permute(0, 2, 1)

            # cal loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
            # mse_loss = torch.sum((prediction - original_point_cloud)**2)
            cd_loss, _ = chamfer_distance(prediction, original_point_cloud)
            # val_loss = kl_loss + mse_loss
            val_loss = kl_loss + 1e7*cd_loss

            # sum train loss
            sum_val_loss += val_loss

    sum_val_loss /= len(dataloader)
    return sum_val_loss
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # prepare options
    parser = make_parser()
    args = parser.parse_args()

    # make path of save params
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    dt_now = datetime.datetime.now()
    save_date = str(dt_now.month) + str(dt_now.day) + "-" + str(dt_now.hour) + "-" +str(dt_now.minute)
    if not os.path.exists(os.path.join(args.save_dir, str(dt_now.year))):
        os.mkdir(os.path.join(args.save_dir, str(dt_now.year)))

    save_dir = os.path.join(args.save_dir, str(dt_now.year), save_date)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        with open(os.path.join(save_dir, "conditions.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get subset id
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
            else:
                print("You should check subset name!")
                exit()

    train_points_path_list, train_subset_name_list = get_item_from_json(args.dataset_dir, "train", subset_id)
    val_points_path_list, val_subset_name_list = get_item_from_json(args.dataset_dir, "val", subset_id)
    # make dataloader
    train_dataset = MakeDataset(points_path_list=train_points_path_list,
                                subset_name_list=train_subset_name_list, device=args.device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch,
                                  shuffle=True, drop_last=True,
                                  collate_fn=CollateUpSampling(args.device)) # DataLoader is iterable object.
    val_dataset = MakeDataset(points_path_list=val_points_path_list,
                              subset_name_list=val_subset_name_list, device=args.device)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch,
                                shuffle=True, drop_last=True,
                                collate_fn=CollateUpSampling(args.device)) # DataLoader is iterable object.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # prepare model
    Model = PointVAE(in_dim=3, z_dim=args.z_dim).to(device=args.device)
    optim = torch.optim.Adam(params=Model.parameters(), lr=args.lr)

    # prepare writter
    writter = SummaryWriter()
    torch.autograd.set_detect_anomaly(True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # main loop
    save_normal = os.path.join(save_dir, "normal_parameters.tar")
    save_best = os.path.join(save_dir, "best_parameters.tar")
    best_loss = np.inf
    for epoch in tqdm(range(1, args.epochs+1), desc="main loop"):

        # train and cal loss
        cd_loss, kl_loss, train_loss = train_one_epoch(Model, train_dataloader, optim)
        # train_loss = train_one_epoch(Model, train_dataloader, optim)
        val_loss = val_one_epoch(Model, val_dataloader)

        # sabe time history of data
        writter.add_scalar("cd_loss", cd_loss, epoch)
        # writter.add_scalar("mse_loss", mse_loss, epoch)
        writter.add_scalar("kl_loss", kl_loss, epoch)
        writter.add_scalar("train_loss", train_loss, epoch)
        writter.add_scalar("validation_loss", val_loss, epoch)

        # save normal loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": Model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "loss": train_loss
        }, save_normal)

        # save best loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": Model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": best_loss
            }, save_best)
