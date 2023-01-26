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

        point_feature, mu, log_var, z = self.encoder(x)

        z = z.view(B, self.z_dim, 1).repeat(1, 1, N)
        features = torch.concat([point_feature, z], dim=1)

        out = self.decoder(features)

        return mu, log_var, z, out

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

        return point_feature, mu, log_var, z

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            SharedMLP(64+z_dim, 256),
            SharedMLP(256, 512),
            SharedMLP(512, 1024),
            SharedMLP(1024, 256),
            SharedMLP(256, 64),
            SharedMLP(64, 3),
            nn.Conv1d(3, 3, 1)
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
