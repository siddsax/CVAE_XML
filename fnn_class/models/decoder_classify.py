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

# class decoder(torch.nn.Module):

#     def __init__(self, params):
        
#         super(decoder, self).__init__()
#         self.params = params

#         self.emb_layer = nn.Linear(params.y_dim, params.e_dim, bias=False) 
#         self.emb_layer.weight = torch.nn.Parameter(torch.from_numpy(params.w2v_w).type(params.dtype))
#         self.emb_layer.weight.requires_grad = False
        
#         # self.l0 = nn.Linear(params.y_dim, params.h_dim, bias=True)
#         if(self.params.compress == 0):
#             if(self.params.layer_y):
#                 self.l1 = nn.Linear(params.Z_dim + params.e_dim, params.h_dim, bias=True)
#             else:
#                 self.l1 = nn.Linear(params.Z_dim + params.y_dim, params.h_dim, bias=True)
#         else:
#             self.l1 = nn.Linear(params.Z_dim, params.h_dim, bias=True)

#         self.bn_1 = nn.BatchNorm1d(params.h_dim)    
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(params.h_dim, params.X_dim, bias=True)
#         self.drp_1 = nn.Dropout(.1)

#         self.l3 = nn.Sigmoid()
#         # weights_init(self.l0.weight)
#         weights_init(self.l1.weight)
#         weights_init(self.l2.weight)

#     def forward(self, z, y):
        
#         if(self.params.compress == 0):
#             if(self.params.layer_y):
#                 o = self.emb_layer(y)/torch.sum(y, dim=1).view(-1,1)
#             else:
#                 o = y
#             o = torch.cat((o, z), dim=-1)
#         else:
#             o = z

#         o = self.l1(o)
#         o = self.bn_1(o)
#         o = self.relu(o)
#         o = self.drp_1(o)
#         # ------------------------------
#         o = self.l2(o)
#         o = self.l3(o)
        
#         return o


class decoder(torch.nn.Module):
    
    def __init__(self, params):
        
        super(decoder, self).__init__()
        self.params = params

        self.emb_layer = nn.Linear(params.y_dim, params.e_dim, bias=False) 
        self.emb_layer.weight = torch.nn.Parameter(torch.from_numpy(params.w2v_w).type(params.dtype))
        self.emb_layer.weight.requires_grad = False
        self.w2v_w = torch.from_numpy(params.w2v_w).type(params.dtype)
        # self.l0 = nn.Linear(params.y_dim, params.h_dim, bias=True)
        if(self.params.compress == 0):
            if(self.params.layer_y):
                self.nrm = nn.BatchNorm1d(params.Z_dim + params.e_dim)
                self.l0 = nn.Linear(params.Z_dim + params.e_dim, params.H_dim, bias=True)
            else:
                self.nrm = nn.BatchNorm1d(params.Z_dim + params.y_dim)
                self.l0 = nn.Linear(params.Z_dim + params.y_dim, params.H_dim, bias=True)
        else:
            self.nrm = nn.BatchNorm1d(params.Z_dim)
            self.l0 = nn.Linear(params.Z_dim, params.H_dim, bias=True)
        
        # ==================================================

        self.lx1 = nn.Linear(params.H_dim, 2*params.H_dim, bias=True)
        self.lx2 = nn.Linear(2*params.H_dim, 3*params.H_dim, bias=True)
        self.lx3 = nn.Linear(3*params.H_dim, 4*params.H_dim, bias=True)
        self.lx4 = nn.Linear(4*params.H_dim, 5*params.H_dim, bias=True)
        self.l2 = nn.Linear(5*params.H_dim, params.X_dim, bias=True)

        weights_init(self.lx1.weight)
        weights_init(self.lx2.weight)
        weights_init(self.lx3.weight)
        weights_init(self.lx4.weight)


        self.l3 = nn.Sigmoid()
        weights_init(self.l0.weight)
        weights_init(self.l2.weight)

        
        # ==================================================

    def forward(self, z, y):
        
        if(self.params.compress == 0):
            if(self.params.layer_y):
                # o = self.emb_layer(y)/torch.sum(y, dim=1).view(-1,1)
                o = torch.mm(y, Variable(self.w2v_w))/torch.sum(y, dim=1).view(-1,1)
            else:
                o = y
                # import pdb
                # pdb.set_trace()
            o = torch.cat((o, z), dim=-1)
        else:
            o = z

        o = self.l0(o)
        o = self.lx1(o)
        o = self.lx2(o)
        o = self.lx3(o)
        o = self.lx4(o)
        o = self.l2(o)
        o = self.l3(o)

        return o