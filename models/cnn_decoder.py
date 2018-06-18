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
from torch.nn import Parameter
import subprocess

class cnn_decoder(nn.Module):
    def __init__(self, params):
        super(cnn_decoder, self).__init__()
        boom = -6
        #boom = get_gpu_memory_map(boom)#-6
        self.params = params
        self.out_size = self.params.decoder_kernels[-1][0]
        self.bn_inp = nn.BatchNorm1d(self.params.sequence_length + 1)
        
        
        # self.drp = nn.DataParallel(nn.Dropout(p=params.drop_prob))
        self.drp = nn.Dropout(p=params.drop_prob)
        
        
        #boom = get_gpu_memory_map(boom)#-5
        self.conv_layers = nn.ModuleList()
        #boom = get_gpu_memory_map(boom)#-4
        for layer in range(len(params.decoder_kernels)):
            [out_chan, in_chan, width] = params.decoder_kernels[layer]
            layer = nn.Conv1d(in_chan, out_chan, width,
                         dilation=self.params.decoder_dilations[layer],
                         padding=self.params.decoder_paddings[layer])
            torch.nn.init.xavier_uniform_(layer)
            
            
            # layer = nn.DataParallel(layer)
            
            
            self.conv_layers.append(layer)
            #boom = get_gpu_memory_map(boom)-1#-3
        
        # self.relu = nn.DataParallel(nn.ReLU())
        # self.relu2 = nn.DataParallel(nn.ReLU())
        
        self.bn_1 = nn.BatchNorm1d(self.out_size)
        self.fc = nn.Linear(self.out_size, self.params.vocab_size)
        self.bn_2 = nn.BatchNorm1d(self.params.sequence_length + 1)
        #boom = get_gpu_memory_map(boom)#-2
        torch.nn.init.xavier_uniform_(self.fc)
        
        
        # self.fc = nn.DataParallel(self.fc)
        
        
        #boom = get_gpu_memory_map(boom)#-1
        # self.sigmoid = nn.DataParallel(nn.Sigmoid()) 
        #boom = get_gpu_memory_map(boom)#0
    def forward(self, decoder_input, z, batch_y):
        boom = 1
        [batch_size, seq_len, embed_size] = decoder_input.size()
        #boom = get_gpu_memory_map(boom)#1
        z = torch.cat([z, batch_y], 1)
        #boom = get_gpu_memory_map(boom)#2
        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.Z_dim + self.params.classes)
        #boom = get_gpu_memory_map(boom)#3
        decoder_input = torch.cat([decoder_input, z], 2)
        
        
        decoder_input = self.bn_inp(decoder_input)
        
        
        decoder_input = self.drp(decoder_input)
        #boom = get_gpu_memory_map(boom)#4
        x = decoder_input.transpose(1, 2).contiguous()
        #boom = get_gpu_memory_map(boom)#5
        for layer in range(len(self.params.decoder_kernels)):
            x = self.conv_layers[layer](x)
            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.params.decoder_paddings[layer])].contiguous()
            x = nn.functional.relu(x)
            #boom = get_gpu_memory_map(boom)-1#6

        x = x.transpose(1, 2).contiguous()
        #boom = get_gpu_memory_map(boom)#6
        
        
        x = self.bn_1(x.view(-1, self.out_size))
        # x = x.view(-1, self.out_size)
        
        
        #boom = get_gpu_memory_map(boom)#7
        x = self.fc(x)
        #boom = get_gpu_memory_map(boom)#8
        
        
        x = self.bn_2(x.view(-1, seq_len, self.params.vocab_size))
        
        
        #boom = get_gpu_memory_map(boom)#9
        x = nn.functional.relu(x)
        #boom = get_gpu_memory_map(boom)#10
        return x
