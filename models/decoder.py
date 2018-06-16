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

class decoder(torch.nn.Module):

    def __init__(self, X_dim, y_dim, h_dim, Z_dim):
        
        super(decoder, self).__init__()
        # nn.init.xavier_uniform_.torch.nn.init.xavier_uniform
        self.l0 = nn.Linear(Z_dim + y_dim, h_dim + y_dim, bias=True)
        self.l1 = nn.DataParallel(nn.ReLU())
        self.bn = nn.BatchNorm1d(h_dim + y_dim)

        self.l2 = nn.Linear(h_dim + y_dim, 2*h_dim + y_dim, bias=True)
        self.l3 = nn.DataParallel(nn.ReLU())
        self.bn2 = nn.BatchNorm1d(2*h_dim + y_dim)
        
        self.l4 = nn.Linear(2*h_dim + y_dim, 3*h_dim + y_dim, bias=True)
        self.l5 = nn.DataParallel(nn.ReLU())
        self.bn3 = nn.BatchNorm1d(3*h_dim + y_dim)

        self.l6 = nn.Linear(3*h_dim + y_dim, X_dim, bias=True)
        # self.l7 = nn.ReLU()
        self.l7 = nn.Sigmoid()

        torch.nn.init.xavier_uniform(self.l0.weight)
        torch.nn.init.xavier_uniform(self.l2.weight)
        torch.nn.init.xavier_uniform(self.l4.weight)
        torch.nn.init.xavier_uniform(self.l6.weight)

        self.l0 = nn.DataParallel(self.l0)
        self.l2 = nn.DataParallel(self.l2)
        self.l4 = nn.DataParallel(self.l4)
        self.l6 = nn.DataParallel(self.l6)



    def forward(self, inputs):
        
        o0 = self.l0(inputs)
        o1 = self.l1(o0)
        obn = self.bn(o1)
        
        o2 = self.l2(obn)
        o3 = self.l3(o2)
        obn2 = self.bn2(o3)

        o4 = self.l4(obn2)
        o5 = self.l5(o4)
        obn3 = self.bn3(o5)

        o6 = self.l6(obn3)
        o7 = self.l7(o6)
        
        return o7
