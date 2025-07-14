import torch
import torch.nn as nn
import torch.optim as optim

#张量操作
x = torch.tensor([1.,2.,3.],requires_grad=True)

#自动微分
y = x.sum()
y.backward()
print(x.grad) #梯度值

#数据集与数据加载器
from torch.utils.data import Dataset, DataLoader