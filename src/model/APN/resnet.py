import torch.nn as nn
from torchvision import models


class ResNet50Backbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        
        # Initial layers (stem)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Residual blocks
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels -> for APN
        self.layer4 = resnet.layer4  # 2048 channels -> for APAN
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        
        return {
            'layer1': layer1_out,
            'layer2': layer2_out,
            'layer3': layer3_out,  # For APN branch
            'layer4': layer4_out   # For APAN branch
        }
