import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, in_ch, squeeze_ch, expand_ch):
        super().__init__()

        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, 1)
        self.expand1 = nn.Conv2d(squeeze_ch, expand_ch, 1)
        self.expand3 = nn.Conv2d(squeeze_ch, expand_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return torch.cat([
            F.relu(self.expand1(x)),
            F.relu(self.expand3(x))
        ], dim=1)

class SmallSqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Conv2d(1, 32, 3, padding=1)

        self.fire1 = FireModule(32, 16, 32)
        self.fire2 = FireModule(64, 16, 64)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.fire1(x)
        x = self.fire2(x)
        return self.head(x)
