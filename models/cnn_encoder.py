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
import subprocess

def get_gpu_memory_map(boom):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print("In Encoder:: Print: {0}; Mem(1): {1}; Mem(2): {2}; Mem(3): {3}; Mem(4): {4}".format( boom, gpu_memory_map[0], \
    gpu_memory_map[1], gpu_memory_map[2], gpu_memory_map[3]))
    return boom+1


def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

def weights_init(m):
    # if isinstance(m, nn.Conv1d):
    #     torch.nn.init.xavier_uniform(m.weight.data)
    #     torch.nn.init.xavier_uniform(m.bias.data)
    # else:
    torch.nn.init.xavier_uniform_(m.weight.data)

class cnn_encoder(torch.nn.Module):
    
    def __init__(self, args):
        boom = -6
        # Model Hyperparameters
        super(cnn_encoder, self).__init__()
        
        # Model Hyperparameters
        self.params = args
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
        #boom = get_gpu_memory_map(boom)#-6
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        #boom = get_gpu_memory_map(boom)#-5
        fin_l_out_size = 0

        self.bn_in = nn.BatchNorm1d(args.sequence_length)
        
        
        # self.drp0 = nn.DataParallel(nn.Dropout(p=.25))
        self.drp0 = nn.Dropout(p=.25)
        
        
        # self.relu = nn.DataParallel(nn.ReLU())
        #boom = get_gpu_memory_map(boom)#-4
        for fsz in self.filter_sizes:
            l_out_size = out_size(self.sequence_length, fsz, stride=2)
            pool_size = l_out_size // self.pooling_units

            # l_conv = nn.Conv1d(self.embedding_dim + args.classes, self.num_filters, fsz, stride=2)
            l_conv = nn.Conv1d(self.embedding_dim, self.num_filters, fsz, stride=2)
            weights_init(l_conv)
            
            
            # l_conv = nn.DataParallel(l_conv)
            
            
            if self.pooling_type == 'average':
                
                
                # l_pool = nn.DataParallel(nn.AvgPool1d(pool_size, stride=None, count_include_pad=True))
                l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
                
                
                fin_l_out_size += (int((l_out_size - pool_size)/pool_size) + 1)*self.num_filters
            elif self.pooling_type == 'max':


                # l_pool = nn.DataParallel(nn.MaxPool1d(2, stride=1))
                l_pool = nn.MaxPool1d(2, stride=1)


                fin_l_out_size += (int(l_out_size*self.num_filters - 2) + 1)

            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)
            #boom = get_gpu_memory_map(boom) - 1#-3
        self.bn1 = nn.BatchNorm1d(fin_l_out_size)
        # self.fc = nn.Linear(fin_l_out_size + args.classes, fin_l_out_size, bias=True)
        # self.bn2 = nn.BatchNorm1d(fin_l_out_size)
        #boom = get_gpu_memory_map(boom)#-2
        self.mu = nn.Linear(fin_l_out_size, self.Z_dim, bias=True)
        self.var = nn.Linear(fin_l_out_size, self.Z_dim, bias=True)
        #boom = get_gpu_memory_map(boom) #-1
        # weights_init(self.fc)
        weights_init(self.var)
        weights_init(self.mu)
        # self.fc = nn.DataParallel(self.fc)
        
        
        # self.mu = nn.DataParallel(self.mu)
        # self.var = nn.DataParallel(self.var)
        
        
        #boom = get_gpu_memory_map(boom) #0
    def forward(self, inputs, batch_y):
        boom = 1
        # [batch_size, seq_len, embed_size] = inputs.size()
        # batch_y = torch.cat([batch_y] * seq_len, 1).view(batch_size, seq_len, self.params.classes)
        # inputs = torch.cat([inputs, batch_y], 2)
        #boom = get_gpu_memory_map(boom)#1
        
        
        o0 = self.drp0(self.bn_in(inputs)) 
        # o0 = self.drp0(inputs) 
        
        
        #boom = get_gpu_memory_map(boom)#2
        conv_out = []
        k = 0
        for i in range(len(self.filter_sizes)):

            o = o0.permute(0,2,1)
            o = self.conv_layers[i](o)
            o = nn.functional.relu(o)
            o = o.view(o.shape[0], 1, o.shape[1]*o.shape[2])
            o = self.pool_layers[i](o)
            o = o.view(o.shape[0],-1)
            conv_out.append(o)
            #boom = get_gpu_memory_map(boom)-1#3
            del o
            #boom = get_gpu_memory_map(boom)-1#3
        del o0
        #boom = get_gpu_memory_map(boom)#3
        if len(self.filter_sizes)>1:
            o = torch.cat(conv_out,1)
        else:
            o = conv_out[0]
        #boom = get_gpu_memory_map(boom)#4
        del conv_out
        #boom = get_gpu_memory_map(boom)#5
        
        
        o = self.bn1(o)
        
        
        #boom = get_gpu_memory_map(boom)#6
        # o = self.fc(torch.cat([batch_y, o], 1))
        # o = self.bn2(o)
        # o1 = self.mu(o)
        # o2 = self.var(o)
        # del o
        
        return self.mu(o),self.var(o)
