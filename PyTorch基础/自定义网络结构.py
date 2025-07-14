import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义图像预处理流程
transform = transforms.ToTensor()  # 转为Tensor，自动缩放到[0,1]

# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),             # 展平 28x28 图像成 784 向量
            nn.Linear(784, 128),      # 全连接层：784 → 128
            nn.ReLU(),                # 激活函数
            nn.Linear(128, 64),       # 128 → 64
            nn.ReLU(),
            nn.Linear(64, 10)         # 输出层：10 个类别
        )

    def forward(self, x):
        return self.model(x)

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()                 # 多分类交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

for epoch in range(5):  # 训练 5 个 epoch
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)                # 前向传播
        loss = criterion(outputs, labels)      # 计算损失

        optimizer.zero_grad()                  # 梯度清零
        loss.backward()                        # 反向传播
        optimizer.step()                       # 更新参数

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

correct = 0
total = 0
model.eval()  # 进入评估模式
with torch.no_grad():  # 关闭梯度计算，提高效率
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total:.4f}")
