import torch
import torch.nn as nn
from module import SharedMLP, LinearMLP

class PointVAE(nn.Module):
    def __init__(self, in_dim, num_out_points, z_dim):
        super().__init__()
        self.num_out_points = num_out_points
        self.encoder = Encoder(in_dim, z_dim)
        self.decoder = Decoder(z_dim, num_out_points)

    def forward(self, x):
        B, _, _ = x.shape

        mu, log_var, z = self.encoder(x)
        out = self.decoder(z)

        out_point_cloud = out.view(B, 3, self.num_out_points)
        return z, out_point_cloud

class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.MLP1 = nn.Sequential(
            SharedMLP(in_dim, 64),
            SharedMLP(64, 64)
        )
        self.MLP2 = nn.Sequential(
            SharedMLP(64, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024)
        )
        self.fc_mu = nn.Sequential(
            LinearMLP(1024, 512),
            LinearMLP(512, 256),
            LinearMLP(256, z_dim),
            nn.Linear(z_dim, z_dim)
        )
        self.fc_var = nn.Sequential(
            LinearMLP(1024, 512),
            LinearMLP(512, 256),
            LinearMLP(256, z_dim),
            nn.Linear(z_dim, z_dim)
        )

    def forward(self, x):
        device = x.device

        # get point feature
        point_feature = self.MLP1(x)

        # get global feature
        global_feature = self.MLP2(point_feature)
        global_feature = torch.max(global_feature, dim=2)[0]

        # get mean and variance
        mu = self.fc_mu(global_feature)
        log_var = self.fc_var(global_feature)

        # reparametrization tric
        eps = torch.randn_like(torch.exp(log_var), device=device)
        z = mu + torch.exp(0.5*log_var)*eps

        return mu, log_var, z

class Decoder(nn.Module):
    def __init__(self, z_dim, num_out_points):
        super().__init__()
        self.fc = nn.Sequential(
            LinearMLP(z_dim, 256),
            LinearMLP(256, 512),
            LinearMLP(512, 1024),
            LinearMLP(1024, 3*num_out_points),
            nn.Linear(3*num_out_points, 3*num_out_points)
        )

    def forward(self, x):
        out = self.fc(x)
        return out

if __name__ == "__main__":
    x = torch.randn(100, 3, 90)
    Net = PointVAE(in_dim=3, num_out_points=1000, z_dim=64)
    z, out = Net(x)

    print(z.shape)
    print(out.shape)
