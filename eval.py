import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
import pandas as pd
from options import make_parser
from model import PointVAE
from data import *

def export_ply(dir_path, file_name, type, point_cloud):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    path = os.path.join(dir_path, type, str(file_name)+".ply")
    o3d.io.write_point_cloud(path, pc)

def eval(model, dataloader, save_dir):
    model.eval()

    feature_df = pd.DataFrame(np.zeros((len_dataset, args.z_dim)), index=np.arange(1, len_dataset+1),
                              columns=np.arange(1, args._dim+1))
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="eval")):
            original_point_cloud = data[0]
            _ = data[1]
            subset_name = data[2]

            # get prediction
            _, _, z, prediction = model(original_point_cloud)

            z = z.detach().cpu().numpy()
            z = z.reshape(args.z_dim)
            feature_df.loc[i+1,:] = z

            original_point_cloud = original_point_cloud.detach().cpu().numpy()
            print(original_point_cloud.shape)
            print(prediction.shape)
            print(subset_name)
            exit


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get options
    parser = make_parser()
    args = parser.parse_args()

    result_dir = os.path.join(args.result_dir, args.subset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    # make eval dataloader
    eval_dataset = MakeDataset(dataset_path=args.dataset_dir, eval="test",
                                subset_id=subset_id, device=args.device)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1,
                                  collate_fn=CollateUpSampling(args.device)) # DataLoader is iterable object.
    len_dataset = len(eval_dataset)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Model = PointVAE(in_dim=3, z_dim=args.z_dim).to(device=args.device)

    # load tar
    tar_path = os.path.join("./checkpoint", args.year, args.date, args.type+"_parameters.tar")
    train_tar = torch.load(tar_path)

    result_dir = 'result'
    result_txt = os.path.join(result_dir, 'result.txt')
    with open(result_txt, 'w') as f:
        f.write('train_data: {}\n'.format(args.date))
        f.write('epoch: {}\n'.format(train_tar['epoch']))
        f.write('loss : {}\n'.format(train_tar['loss']))

    Model.load_state_dict(train_tar["model_state_dict"])
    eval(Model, eval_dataloader, result_dir)
