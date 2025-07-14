import torch.nn as nn
from models.residual_block import ResidualBlock

class ResNetCIFAR(nn.Module):
    def __init__(self):
        super(ResNetCIFAR, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.resblock1 = ResidualBlock(16, 16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.resblock2 = ResidualBlock(32, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.resblock1(out)
        out = self.layer2(out)
        out = self.resblock2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
