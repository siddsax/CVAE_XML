import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from weights_init import weights_init
# from pycrayon import CrayonClient

class decoder(torch.nn.Module):

    def __init__(self, params):
        
        super(decoder, self).__init__()
        self.l0 = nn.Linear(params.y_dim, params.h_dim, bias=True)
        self.l1 = nn.Linear(params.Z_dim + params.h_dim, params.h_dim, bias=True)
        self.relu = nn.ReLU()
        # self.drp_5 = nn.Dropout(.5)
        # self.bn = nn.BatchNorm1d(params.h_dim)
        # self.bn_1 = nn.BatchNorm1d(params.h_dim)
        self.l2 = nn.Linear(params.h_dim, params.X_dim, bias=True)
        self.drp_1 = nn.Dropout(.1)

        # if(params.fin_layer == "Sigmoid"):
        # elif(params.fin_layer == "ReLU"):
        self.l3 = nn.Sigmoid()
        # elif(params.fin_layer == "None"):
        #     self.l3 = ""

        weights_init(self.l0.weight)
        weights_init(self.l1.weight)
        weights_init(self.l2.weight)

    def forward(self, z, y):
        
        o = self.l0(y)
        # o = self.bn(o)
        o = self.relu(o)
        # o = self.drp_1(o)
        #--------------------------------
        o = torch.cat((o, z), dim=-1)
        o = self.l1(o)
        # o = self.bn_1(o)
        o = self.relu(o)
        # o = self.drp_1(o)
        # ------------------------------
        o = self.l2(o)
        # o = self.l3(o)
        
        return o
