import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
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

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="eval")):
            original_point_cloud = data[0]

            # get prediction
            z, prediction = model(original_point_cloud)


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get options
    parser = make_parser()
    args = parser.parse_args()
