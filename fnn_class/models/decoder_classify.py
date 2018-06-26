import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
# from pycrayon import CrayonClient

class decoder(torch.nn.Module):

    def __init__(self, params):
        
        super(decoder, self).__init__()
        self.l0 = nn.Linear(params.Z_dim, params.h_dim, bias=True)
        self.l1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(params.h_dim)
        self.l2 = nn.Linear(params.h_dim, params.y_dim, bias=True)

        if(params.fin_layer == "Sigmoid"):
            self.l3 = nn.Sigmoid()
        elif(params.fin_layer == "ReLU"):
            self.l3 = nn.ReLU()
        elif(params.fin_layer == "None"):
            self.l3 = ""

        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)

        # self.cc = CrayonClient(hostname="localhost")
        # self.summary = self.cc.create_experiment(type(self).__name__)

    def forward(self, inputs):
        
        o = self.l0(inputs)
        o = self.l1(o)
        o = self.bn(o)
        o = self.l2(o)
        if(type(self.l3)!=str):
            o = self.l3(o)
        
        return o
