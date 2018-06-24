import os
import sys
import torch
import random

import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
sys.path.insert(0, '../utils')
from futils import *
from w2v import load_word2vec
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.gridspec as gridspec
import subprocess

def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1

class cnn_encoder(torch.nn.Module):
    
    def __init__(self, args):
        super(cnn_encoder, self).__init__()
        self.args = args
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        fin_l_out_size = 0
        self.bn_in = nn.BatchNorm1d(args.sequence_length)
        self.drp0 = nn.Dropout(p=.25)

        for fsz in args.filter_sizes:
            l_out_size = out_size(args.sequence_length, fsz, stride=2)
            pool_size = l_out_size // args.pooling_units
            l_conv = nn.Conv1d(args.embedding_dim, args.num_filters, fsz, stride=2)
            torch.nn.init.xavier_uniform_(l_conv.weight)
            if args.pooling_type == 'average':
                l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
                fin_l_out_size += (int((l_out_size - pool_size)/pool_size) + 1)*args.num_filters
            elif args.pooling_type == 'max':
                l_pool = nn.MaxPool1d(2, stride=1)
                fin_l_out_size += (int(l_out_size*args.num_filters - 2) + 1)
            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)

        self.bn1 = nn.BatchNorm1d(fin_l_out_size)
        self.mu = nn.Linear(fin_l_out_size, args.Z_dim, bias=True)
        self.var = nn.Linear(fin_l_out_size, args.Z_dim, bias=True)
        torch.nn.init.xavier_uniform_(self.var.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight)

    def forward(self, inputs, batch_y):
        o0 = self.drp0(self.bn_in(inputs)) 
        conv_out = []
        k = 0

        for i in range(len(self.args.filter_sizes)):
            o = o0.permute(0,2,1)
            o = self.conv_layers[i](o)
            o = nn.functional.relu(o)
            o = o.view(o.shape[0], 1, o.shape[1]*o.shape[2])
            o = self.pool_layers[i](o)
            o = o.view(o.shape[0],-1)
            conv_out.append(o)
            del o
        del o0
        if len(self.args.filter_sizes)>1:
            o = torch.cat(conv_out,1)
        else:
            o = conv_out[0]
        del conv_out
        o = self.bn1(o)
        return self.mu(o),self.var(o)
