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
        self.l1 = nn.Linear(params.Z_dim, params.h_dim, bias=True)
        self.l2 = nn.ReLU()
        self.bn = nn.BatchNorm1d(params.h_dim)
        self.drp = nn.Dropout(.5)
        self.l3 = nn.Linear(params.h_dim, params.y_dim, bias=True)

        # if(params.fin_layer == "Sigmoid"):
        #     self.l3 = nn.Sigmoid()
        # elif(params.fin_layer == "ReLU"):
        #     self.l3 = nn.ReLU()
        # elif(params.fin_layer == "None"):
        #     self.l3 = ""

        weights_init(self.l0.weight)
        weights_init(self.l1.weight)
        weights_init(self.l3.weight)

    def forward(self, z, y):
        
        o = self.l0(y)
        o = torch.cat((o, z), dim=-1)
        o = self.drp(o)
        o = self.l2(o)
        o = self.l1(o)
        o = self.drp(o)
        o = self.bn(o)
        o = self.l2(o)
        # if(type(self.l3)!=str):
        #     o = self.l3(o)
        
        return o
