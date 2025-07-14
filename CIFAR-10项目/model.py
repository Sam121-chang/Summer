"""# 自定义 CNN 网络结构定义文件"""

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：输入3通道(RGB)，输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 加 BatchNorm

        # 第二层卷积：32 → 64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)  # 池化层：2x2 降采样

        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 展平后全连接层
        self.fc2 = nn.Linear(128, 10)  # 最终输出10类

    def forward(self, x):
        # 第一卷积+BN+激活+池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 第二卷积+BN+激活+池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.view(-1, 64 * 8 * 8)  # 展平为1维输入
        x = F.relu(self.fc1(x))    # 第一个全连接层
        x = self.fc2(x)            # 输出层，不加 softmax（交叉熵损失函数内部包含）

        return x
