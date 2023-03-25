import torch
import torch.nn as nn
from torch.nn import init

import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, in_size, h1_size, h2_size, h3_size, h4_size,  out_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_size, h1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(h2_size, h3_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(h3_size, h4_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(h4_size, out_size)
                
    def forward(self, x):
        
        if x.ndim < 2:
            x = x.unsqueeze(0)
        
        out = self.fc1(x)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        
        out = self.fc4(out)
        out = self.relu4(out)
        
        out = self.fc5(out)
        
        
        return out