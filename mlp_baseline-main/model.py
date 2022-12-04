import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F


## 残差模块
class ResidualBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(nn.Linear(self.input_size, self.output_size), nn.BatchNorm1d(self.output_size), nn.ReLU(), nn.Linear(self.output_size, self.output_size), nn.BatchNorm1d(self.output_size))
        if self.input_size == self.output_size:
            self.l = None
        else:
            self.l = nn.Linear(self.input_size, self.output_size)
          
    def forward(self, x):
        y = self.net(x)
        if self.input_size != self.output_size:
            x = self.l(x) 
        return F.relu(x + y)

## 降采样模块    
class DownBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(nn.Linear(self.input_size, self.output_size), nn.BatchNorm1d(self.output_size), nn.ReLU())
    
    def forward(self, x):
        return self.net(x)

## mlp模型
class mlp(nn.Module):
    def __init__(self, input_size=359, output_size=1):
        super().__init__()
        self.block = nn.Sequential(DownBlock(input_size, 512), ResidualBlockMLP(512, 256), DownBlock(256,64), ResidualBlockMLP(64,32), nn.Linear(32, 1))

    def forward(self, x):
        y = self.block(x)
        return y

