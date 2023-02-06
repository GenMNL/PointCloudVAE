import torch
import torch.nn as nn
from module import SharedMLP, LinearMLP

class PointVAE(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(in_dim, z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        B, C, N = x.shape

        # ==================
        # global_feature = self.encoder(x)
        # out = self.decoder(global_feature)
        # out = out.view(B, 2000, C)
        # return global_feature, out
        # ==================
        point_feature, mu, log_var, z = self.encoder(x)

        out = self.decoder(z)
        out = out.view(B, 2000, C)
        return mu, log_var, z, out

        # for point-wise decoder
        z_clone = z.clone().detach()
        z = z.view(B, self.z_dim, 1).repeat(1, 1, N)
        features = torch.concat([point_feature, z], dim=1)

        out = self.decoder(features)
        out = out.view(B, C, N)

        return mu, log_var, z_clone, out


class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.MLP1 = nn.Sequential(
            SharedMLP(in_dim, 64)
        )
        self.MLP2 = nn.Sequential(
            SharedMLP(64, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 256),
            SharedMLP(256, 512),
        )
        self.fc_mu = nn.Sequential(
            LinearMLP(512, z_dim),
            nn.Linear(z_dim, z_dim)
        )
        self.fc_var = nn.Sequential(
            LinearMLP(512, z_dim),
            nn.Linear(z_dim, z_dim)
        )

        self.fc_global = nn.Sequential(
            LinearMLP(512, z_dim),
            nn.Linear(z_dim, z_dim)
        )

    def forward(self, x):
        device = x.device

        # get point feature
        point_feature = self.MLP1(x)

        # get global feature
        global_feature = self.MLP2(point_feature)
        global_feature = torch.max(global_feature, dim=2)[0]

        # ==================
        # global_feature = self.fc_global(global_feature)
        # return global_feature
        # ==================

        # get mean and variance
        mu = self.fc_mu(global_feature)
        log_var = self.fc_var(global_feature)

        # reparametrization tric
        eps = torch.randn_like(torch.exp(log_var), device=device)
        z = mu + torch.exp(0.5*log_var)*eps

        return point_feature, mu, log_var, z

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.SharedMLP = nn.Sequential(
            # SharedMLP(64+z_dim, 256),
            # SharedMLP(256, 512),
            # SharedMLP(512, 1024),
            # SharedMLP(1024, 256),
            # SharedMLP(256, 64),
            # SharedMLP(64, 3),
            # nn.Conv1d(3, 3, 1)
            SharedMLP(1+z_dim, 64),
            SharedMLP(64, 32),
            SharedMLP(32, 16),
            SharedMLP(16, 3),
            nn.Conv1d(3, 3, 1)
        )
        self.fc = nn.Sequential(
            LinearMLP(z_dim, 128),
            LinearMLP(128, 512),
            LinearMLP(512, 1024),
            LinearMLP(1024, 6000),
            nn.Linear(6000, 6000)
        )

    def forward(self, x):
        out = self.fc(x)
        return out

if __name__ == "__main__":
    x = torch.randn(100, 3, 90)
    Net = PointVAE(in_dim=3, z_dim=256)
    mu, log_var, z, out = Net(x)

    print(z.shape)
    print(out.shape)
