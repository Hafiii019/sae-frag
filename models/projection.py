import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return x