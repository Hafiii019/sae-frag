import torch
import torch.nn as nn
from models.backbone import ResNet101Backbone
from models.fpn import FPN
from models.safe import SAFE

class MultiViewBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet101Backbone(pretrained=True)
        self.fpn = FPN(out_channels=256)
        self.safe = SAFE(embed_dim=256, num_heads=8)

    def forward(self, x):
        B, V, C, H, W = x.shape

        view1 = x[:, 0]
        view2 = x[:, 1]

        c2_1, c3_1, c4_1, c5_1 = self.backbone(view1)
        c2_2, c3_2, c4_2, c5_2 = self.backbone(view2)

        p2_1, p3_1, p4_1, p5_1 = self.fpn(c2_1, c3_1, c4_1, c5_1)
        p2_2, p3_2, p4_2, p5_2 = self.fpn(c2_2, c3_2, c4_2, c5_2)

        enhanced_1 = self.safe(c5_1, p5_1)
        enhanced_2 = self.safe(c5_2, p5_2)

        fused = enhanced_1 + enhanced_2

        return fused