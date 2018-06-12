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

class embedding_layer(torch.nn.Module):

    def __init__(self, params, embedding_weights):
        super(embedding_layer, self).__init__()
        self.l = nn.Embedding(params.vocab_size, params.embedding_dim)
        if params.model_variation == 'pretrain':
            self.l.weight.data.copy_(torch.from_numpy(embedding_weights))

            # .weight = nn.Parameter(embedding_weights)
        self.l = nn.DataParallel(self.l)
    
    def forward(self, inputs):

        o = self.l(inputs)
      
        return o
