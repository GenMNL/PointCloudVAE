import torch
from pytorch3d.loss import chamfer_distance

source_cloud = torch.randn(2, 100, 3).cuda()
target_cloud = torch.randn(2, 50, 3).cuda()


pytorch3D_CD, _ = chamfer_distance(source_cloud, target_cloud)
print(f"pytorch3D's library is :{pytorch3D_CD}")
