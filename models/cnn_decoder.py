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
from torch.nn import Parameter

def weights_init(m):
    # if isinstance(m, nn.Conv1d):
    #     torch.nn.init.xavier_uniform(m.weight.data)
    #     torch.nn.init.xavier_uniform(m.bias.data)
    # else:
    torch.nn.init.xavier_uniform(m.weight.data)

class cnn_decoder(nn.Module):
    def __init__(self, params):
        super(cnn_decoder, self).__init__()

        self.params = params
        self.out_size = self.params.decoder_kernels[-1][0]

        self.drp = nn.DataParallel(nn.Dropout(p=params.drop_prob))

        self.conv_layers = nn.ModuleList()
        for layer in range(len(params.decoder_kernels)):
            [out_chan, in_chan, width] = params.decoder_kernels[layer]
            layer = nn.Conv1d(in_chan, out_chan, width,
                         dilation=self.params.decoder_dilations[layer],
                         padding=self.params.decoder_paddings[layer])
            weights_init(layer)
            layer = nn.DataParallel(layer)
            self.conv_layers.append(layer)
        
        self.relu = nn.DataParallel(nn.ReLU())
        
        self.fc = nn.Linear(self.out_size, self.params.vocab_size)
        weights_init(self.fc)
        self.fc = nn.DataParallel(self.fc)
        self.sigmoid = nn.DataParallel(nn.Sigmoid()) 
    
    def forward(self, decoder_input, z):

        [batch_size, seq_len, embed_size] = decoder_input.size()
        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.Z_dim)
        decoder_input = torch.cat([decoder_input, z], 2)
        decoder_input = self.drp(decoder_input)

        x = decoder_input.transpose(1, 2).contiguous()
        for layer in range(len(self.params.decoder_kernels)):
            x = self.conv_layers[layer](x)
            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.params.decoder_paddings[layer])].contiguous()
            x = self.relu(x)


        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, self.out_size)
        x = self.fc(x)
        x = x.view(-1, seq_len, self.params.vocab_size)
        # print(x.shape)
        x = self.sigmoid(x)
        # print(x)
        # sys.exit()
        return x
