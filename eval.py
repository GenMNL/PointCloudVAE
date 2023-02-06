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

def resize(normal_tensor, max_values, min_values):
    resize_tensor = normal_tensor*(max_values - min_values)
    resize_tensor = resize_tensor + min_values
    return resize_tensor

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
            max_values = data[3]
            min_values = data[4]

            _, C, N = original_point_cloud.shape

            # get prediction
            # _, _, z, prediction = model(original_point_cloud)
            z, prediction = model(original_point_cloud)

            original_point_cloud = resize(original_point_cloud, max_values, min_values)
            prediction = resize(prediction.permute(0, 2, 1), max_values, min_values).permute(0, 2, 1)

            z = z.detach().cpu().numpy()
            z = z.reshape(args.z_dim)
            feature_df.loc[i+1,:] = z
            label_id = convert_subset_name_id(subset_name)
            labels[0, i] = int(label_id)

            original_point_cloud = original_point_cloud.detach().cpu().numpy()
            original_point_cloud = original_point_cloud.reshape(C, N).T
            prediction = prediction.detach().cpu().numpy()
            prediction = prediction.reshape(2000, C)

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

    test_points_path_list, test_subset_name_list = get_item_from_json(args.dataset_dir, "test", subset_id)
    # make eval dataloader
    test_dataset = MakeDataset(points_path_list=test_points_path_list,
                               subset_name_list=test_subset_name_list, device=args.device)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1,
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
    eval(Model, test_dataloader, args.result_dir)
