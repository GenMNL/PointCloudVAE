import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
import pandas as pd
from options import make_parser
from model import PointVAE
from data import *

def convert_subset_name_id(subset_name):
    subset_info_ = os.path.join(args.dataset_dir, "synsetoffset2category.txt")

    with open(subset_info_, "r") as f:
        for idx, subset in enumerate(f):
            name, _ = subset.split()
            if subset_name == name:
                label_id = idx
                break
    return label_id

def export_ply(dir_path, file_name, type, point_cloud):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    path = os.path.join(dir_path, type, str(file_name)+".ply")
    o3d.io.write_point_cloud(path, pc)

def eval(model, dataloader, save_dir):
    model.eval()

    len_dataset = len(dataloader)
    feature_df = pd.DataFrame(np.zeros((len_dataset, args.z_dim)), index=np.arange(1, len_dataset+1),
                              columns=np.arange(1, args.z_dim+1))
    labels = np.zeros((1, len_dataset), dtype=int)
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="eval")):
            # load data
            original_point_cloud = data[0]
            _ = data[1]
            subset_name = data[2]

            _, C, N = original_point_cloud.shape

            # get prediction
            _, _, z, prediction = model(original_point_cloud)

            z = z.detach().cpu().numpy()
            z = z.reshape(args.z_dim)
            feature_df.loc[i+1,:] = z
            label_id = convert_subset_name_id(subset_name)
            labels[0, i] = int(label_id)

            original_point_cloud = original_point_cloud.detach().cpu().numpy()
            original_point_cloud = original_point_cloud.reshape(C, N).T
            prediction = prediction.detach().cpu().numpy()
            prediction = prediction.reshape(C, N).T

            subset_save_dir = os.path.join(save_dir, subset_name[0])
            if not os.path.exists(subset_save_dir):
                os.makedirs(subset_save_dir)
                os.makedirs(os.path.join(subset_save_dir, "ground_truth"))
                os.makedirs(os.path.join(subset_save_dir, "prediction"))

            export_ply(subset_save_dir, i, "ground_truth", original_point_cloud)
            export_ply(subset_save_dir, i, "prediction", prediction)
    
    feature_path = os.path.join(save_dir, "emb.csv")
    feature_df.to_csv(feature_path)
    labels_save_path = os.path.join(save_dir, 'labels.csv')
    np.savetxt(labels_save_path, labels, delimiter=',')


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get options
    parser = make_parser()
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Model = PointVAE(in_dim=3, z_dim=args.z_dim).to(device=args.device)

    # load tar
    tar_path = os.path.join("./checkpoint", args.year, args.date, args.type+"_parameters.tar")
    train_tar = torch.load(tar_path)

    result_txt = os.path.join(args.result_dir, 'result.txt')
    with open(result_txt, 'w') as f:
        f.write('train_data: {}\n'.format(args.date))
        f.write('epoch: {}\n'.format(train_tar['epoch']))
        f.write('loss : {}\n'.format(train_tar['loss']))

    Model.load_state_dict(train_tar["model_state_dict"])
    eval(Model, eval_dataloader, args.result_dir)
