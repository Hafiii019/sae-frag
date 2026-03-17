import torch
import torch.nn as nn
from models.multiview_backbone import MultiViewBackbone


class SAEImageClassifier(nn.Module):

    def __init__(self, num_classes=14):
        super().__init__()

        # SAEnet backbone (ResNet101 + FPN + SAFE + multi-view fusion)
        self.backbone = MultiViewBackbone()

        # Global average and max pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),   # 256 avg + 256 max = 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images):

        # (B, 2, 3, 224, 224)
        features = self.backbone(images)   # (B, 256, 7, 7)

        avg_feat = self.avgpool(features).flatten(1)  # (B, 256)
        max_feat = self.maxpool(features).flatten(1)  # (B, 256)

        fused = torch.cat([avg_feat, max_feat], dim=1)  # (B, 512)

        logits = self.classifier(fused)

        return logits