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

class decoder(torch.nn.Module):
    
    def __init__(self, params):
        
        super(decoder, self).__init__()
        self.params = params

        self.emb_layer = nn.Linear(params.y_dim, params.e_dim, bias=False) 
        self.emb_layer.weight = torch.nn.Parameter(torch.from_numpy(params.w2v_w).type(params.dtype))
        self.emb_layer.weight.requires_grad = False
        self.w2v_w = torch.from_numpy(params.w2v_w).type(params.dtype)
        if(self.params.layer_y):
            self.bn_cat = nn.BatchNorm1d(params.Z_dim + params.e_dim)
            self.l0 = nn.Linear(params.Z_dim + params.e_dim, params.H_dim, bias=True)
        else:
            self.bn_cat = nn.BatchNorm1d(params.Z_dim + params.y_dim)
            self.l0 = nn.Linear(params.Z_dim + params.y_dim, params.H_dim, bias=True)
        self.bn_l0 = nn.BatchNorm1d(params.H_dim)
        # ==================================================
        self.l2 = nn.Linear(params.H_dim, params.X_dim, bias=True)
        self.relu = nn.ReLU()
        self.drp_5 = nn.Dropout(.5)
        self.drp_1 = nn.Dropout(.1)
        self.l3 = nn.Sigmoid()
        weights_init(self.l0.weight)
        weights_init(self.l2.weight)
        # ==================================================

    def forward(self, z, y):
        
        if(self.params.layer_y):
            y_sum = torch.sum(y, dim=1).view(-1,1)
            divisor = torch.max(y_sum, Variable(torch.ones(y_sum.shape).type(self.params.dtype)))
            y = torch.mm(y, Variable(self.w2v_w))
            o = y/divisor
        else:
            o = y
        o = torch.cat((o, z), dim=-1)
        o = self.bn_cat(o)
        o = self.l0(o)
        o = self.bn_l0(o)
        o = self.relu(o)
        o = self.l2(o)
        o = self.l3(o)

        return o
