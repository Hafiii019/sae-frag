import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()

        # Lateral 1x1 convolutions
        self.lateral_c2 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(2048, out_channels, kernel_size=1)

        # Output 3x3 convolutions
        self.output_c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_c3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_c4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_c5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, c2, c3, c4, c5):
        # Lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        # Output smoothing
        p2 = self.output_c2(p2)
        p3 = self.output_c3(p3)
        p4 = self.output_c4(p4)
        p5 = self.output_c5(p5)

        return p2, p3, p4, p5