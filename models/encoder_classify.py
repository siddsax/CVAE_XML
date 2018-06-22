import torch
import torch.nn as nn
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
        
        self.l0 = nn.Linear(X_dim, h_dim, bias=True)
        self.l1 = nn.DataParallel(nn.ReLU())
        self.bn = nn.BatchNorm1d(h_dim)
        
        # self.l2 = nn.Linear(h_dim, h_dim, bias=True)
        # self.l3 = nn.DataParallel(nn.ReLU())
        # self.bn2 = nn.BatchNorm1d(h_dim)
        
        # self.l4 = nn.Linear(2*h_dim, h_dim, bias=True)
        # self.l5 = nn.DataParallel(nn.ReLU())
        # self.bn3 = nn.BatchNorm1d(h_dim)

        self.mu = nn.Linear(h_dim, Z_dim, bias=True)
        self.var = nn.Linear(h_dim, Z_dim, bias=True)
        
        torch.nn.init.xavier_uniform_(self.l0.weight)
        # torch.nn.init.xavier_uniform_(self.l2.weight)
        # torch.nn.init.xavier_uniform_(self.l4.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.var.weight)
        
        self.l0 = nn.DataParallel(self.l0)
        # self.l2 = nn.DataParallel(self.l2)
        # self.l4 = nn.DataParallel(self.l4)
        self.mu = nn.DataParallel(self.mu)
        self.var = nn.DataParallel(self.var)
        
    def forward(self, inputs):
        
        o0 = self.l0(inputs)
        o1 = self.l1(o0)
        obn = self.bn(o1)

        # o2 = self.l2(obn)
        # o3 = self.l3(o2)
        # obn2 = self.bn2(o3)
        
        # o4 = self.l4(obn2)
        # o5 = self.l5(o4)
        # obn3 = self.bn3(o5)
        
        o6 = self.mu(obn)
        o6_ = self.var(obn)
        # o6 = self.mu(obn3)
        # o6_ = self.var(obn3)
        
        return o6, o6_
