import torch
import torch.nn as nn
import torch.nn.functional as F

from partA.fire import FireModule
from partA.inception import InceptionBlock
from partA.residual import ResidualBlock

'''
Early firemodule: reduce parameters early
Middle inception block: capture multi scale features
Late Residual block: stabilize optimization and improve gradient flow
global average pooling: remove fragile FC
'''


class SuperNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Fire: reduce parameters early
        self.fire = FireModule(
            in_ch=32,
            squeeze_ch=16,
            expand_ch=32       
        )

        # Inception: multi-scale features
        self.inception = InceptionBlock(
            in_ch=64,
            ch1=16,
            ch3=16,
            ch5=16,
            pool_ch=16          
        )

        # Residual: stable deeper processing
        self.residual = ResidualBlock(
            in_channels=64,
            out_channels=128,
            stride=2           
        )

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.fire(x)
        x = self.inception(x)
        x = self.residual(x)
        return self.head(x)
