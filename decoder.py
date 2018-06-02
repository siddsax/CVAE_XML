import torch
# import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable

class decoder(torch.nn.Module):

    def __init__(self, X_dim, y_dim, h_dim, Z_dim):
        
        super(decoder, self).__init__()
        
        self.l0 = torch.nn.Linear(Z_dim + y_dim, h_dim, bias=True)
        self.l1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(h_dim, X_dim, bias=True)
        self.l3 = torch.nn.Sigmoid()
        
    def forward(self, inputs):
        
        o0 = self.l0(inputs)
        o1 = self.l1(o0)
        o2 = self.l2(o1)
        o3 = self.l3(o2)
        
        return o3