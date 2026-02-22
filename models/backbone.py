import torch
import torch.nn as nn
import torchvision.models as models


class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        resnet = models.resnet101(weights=weights)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

    def forward(self, x):
        x = self.layer0(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5