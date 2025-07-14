"""# 工具函数：准确率计算"""

def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)
