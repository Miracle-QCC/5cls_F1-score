import torch
from torch import nn

class Module(nn.Module):
    def __init__(self, input_dim, num_class):
        super(Module, self).__init__()
        self.fc1 = nn.Linear(input_dim,64)
        self.fc2 = nn.Linear(64,64)
        self.cls = nn.Linear(64,num_class)
        self.act = nn.ReLU()
    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        cls = self.cls(out)
        return cls