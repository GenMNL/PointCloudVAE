import open3d as o3d
import os
import numpy as np

if __name__ == "__main__":

    subset = 'Airplane'
    num = 240

    path = os.path.join("result", subset, 'prediction', str(num)+".ply")
    # path = os.path.join("result", subset, 'ground_truth', str(num)+".ply")

    pc = o3d.io.read_point_cloud(path)
    # xyz = np.asarray(pc.points)
    # print(xyz.shape)
    # exit()

    o3d.visualization.draw_geometries([pc])
