"""# 模型训练脚本"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from utils import accuracy

# 将所有主程序逻辑放入该判断内
if __name__ == '__main__':
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据预处理与加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 注意：CIFAR10是3通道，这里可能需要修改为(0.5, 0.5, 0.5)和(0.5, 0.5, 0.5)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # num_workers=2 会启用多进程，需要被 if __name__ == '__main__' 保护
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # 2. 模型初始化
    model = SimpleCNN().to(device)

    # 3. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 模型训练循环
    for epoch in range(5):  # 训练5轮
        model.train()
        running_loss = 0.0
        total_acc = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            total_acc += accuracy(outputs, labels)

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Accuracy: {total_acc/len(trainloader):.4f}")

    print("训练完成 ✅")