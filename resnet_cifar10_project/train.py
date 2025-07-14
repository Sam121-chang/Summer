import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders
from models.resnet_model import ResNetCIFAR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, testloader = get_dataloaders()

model = ResNetCIFAR().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Accuracy: {correct/total:.4f}")
