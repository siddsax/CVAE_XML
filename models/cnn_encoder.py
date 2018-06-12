import os
import sys
import torch
import random

import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
sys.path.insert(0, '../utils')

from w2v import load_word2vec
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.gridspec as gridspec

def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class cnn_encoder(torch.nn.Module):

    def __init__(self, args):
        # Model Hyperparameters
        super(cnn_encoder, self).__init__()
        
        # Model Hyperparameters
        self.sequence_length = args.sequence_length
        self.embedding_dim = args.embedding_dim
        self.filter_sizes = args.filter_sizes
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.pooling_type = args.pooling_type
        self.hidden_dims = args.hidden_dims
        self.Z_dim = args.Z_dim
        # Training Hyperparameters
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs

        # Model variation and w2v pretrain type
        self.vocab_size = args.vocab_size
        # Fix random seed


        self.drp0 = nn.Dropout(p=.25)

        self.conv_layers = []
        self.pool_layers = []

        self.relu = nn.ReLU()
        fin_l_out_size = 0
        print("Input to this layer = ({0},{1},{2})".format(self.batch_size,self.embedding_dim, self.sequence_length ))
        print("#"*50)
        for fsz in self.filter_sizes:
            l_out_size = out_size(self.sequence_length, fsz, stride=2)# l_in, kernel_size, padding=0, dilation=1, stride=1
            pool_size = l_out_size // self.pooling_units
            # print(fin_l_out_size)
            # print(l_out_size)
            l_conv = nn.Conv1d(self.embedding_dim, self.num_filters, fsz, stride=2)
            print(l_out_size*self.num_filters)
            if self.pooling_type == 'average':
                l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
                fin_l_out_size += (int((l_out_size - pool_size)/pool_size) + 1)*self.num_filters
            elif self.pooling_type == 'max':
                l_pool = nn.MaxPool1d(2, stride=1)
                fin_l_out_size += (int(l_out_size*self.num_filters - 2) + 1)
                print((int(l_out_size*self.num_filters - 2) + 1))
                
                print('-'*50)

            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)
        print("*"*50)
        
        # print(fin_l_out_size)
        self.bn = nn.BatchNorm1d(fin_l_out_size)
        self.mu = nn.Linear(fin_l_out_size, self.Z_dim, bias=True)
        self.var = nn.Linear(fin_l_out_size, self.Z_dim, bias=True)

    def forward(self, inputs):

        o0 = self.drp0(inputs)

        conv_out = []
        k = 0
        for i in range(len(self.filter_sizes)):

            o = o0.permute(0,2,1)
            print(o.shape)
            print("#"*50)
            o = self.relu(self.conv_layers[i](o))
            o = o.view(o.shape[0], 1, o.shape[1]*o.shape[2])
            print(o.shape)
            o = self.pool_layers[i](o)
            o = o.view(o.shape[0],-1)
            print(o.shape)
            print('-'*50)
            # k = 
            conv_out.append(o)

        print("+"*50)
        if len(self.filter_sizes)>1:
            conv_out = torch.cat(conv_out,1)
        else:
            conv_out = convs[0]
        
        o = self.bn(conv_out)
        o1 = self.mu(o)
        o2 = self.var(o)
        
        return o1,o2
