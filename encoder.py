import torch
# import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable

class encoder(torch.nn.Module):

    def __init__(self, X_dim, y_dim, h_dim, Z_dim):
        
        super(encoder, self).__init__()
        
        self.l0 = torch.nn.Linear(X_dim + y_dim, h_dim, bias=True)
        self.l1 = torch.nn.ReLU()
        self.mu = torch.nn.Linear(h_dim, Z_dim, bias=True)
        self.var = torch.nn.Linear(h_dim, Z_dim, bias=True)
        
    def forward(self, inputs):
        
        o0 = self.l0(inputs)
        o = self.l1(o0)
        
        o1 = self.mu(o)
        o2 = self.var(o)
        
        return o1, o2