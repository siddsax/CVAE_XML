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

def out_size(l_in, padding, dilation, kernel_size, stride):
    a = l_in + 2*padding - dilation*(kernel_size - 1) -1
    b = a/stride
    return b + 1

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class embedding_layer(torch.nn.Module):

    def __init__(self, params, embedding_weights):
        # Model Hyperparameters
        super(embedding_layer, self).__init__()
        

        self.l = nn.Embedding(params.vocab_size, params.embedding_dim)
        if params.model_variation == 'pretrain':
            self.l.weights = embedding_weights

        # else:
        #     self.l = nn.Embedding(params.vocab_size, params.embedding_dim)


    def forward(self, inputs):

        o = self.l(inputs)
      
        return o
