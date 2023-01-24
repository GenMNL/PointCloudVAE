import torch
import torch.nn as nn

class SharedMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        out = self.main(x)
        return out

class LinearMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.main(x)
        return out

if __name__ == "__main__":
    x = torch.randn(100, 2, 199)
    Net = SharedMLP(2, 10)
    out = Net(x)
    print(x.shape)
    print(out.shape)
