import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, ch1, ch3, ch5, pool_ch):
        super().__init__()

        self.b1 = nn.Conv2d(in_ch, ch1, 1)

        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3, 1),
            nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, padding=1)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5, 1),
            nn.ReLU(),
            nn.Conv2d(ch5, ch5, 5, padding=2)
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_ch, 1)
        )

    def forward(self, x):
        return torch.cat([
            self.b1(x),
            self.b2(x),
            self.b3(x),
            self.b4(x)
        ], dim=1)

class SmallInception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Conv2d(1, 32, 3, padding=1)

        self.inc1 = InceptionBlock(32, 16, 16, 16, 16)
        self.inc2 = InceptionBlock(64, 32, 32, 32, 32)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        return self.head(x)
