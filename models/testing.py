import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import sys
import argparse
from cnn_encoder import cnn_encoder
import torch
import scipy
import numpy as np


parser = argparse.ArgumentParser(description='Process some integers.')
args = parser.parse_args()
args.a = 1
args.sequence_length = 10
args.embedding_dim = 2
args.filter_sizes = [2, 4]
args.num_filters =  3
args.pooling_units = 2
args.pooling_type = 'max'
args.hidden_dims = 2
args.batch_size = 2
args.num_epochs = 2
args.model_variation = 'as'
args.vocab_size = 2
args.Z_dim = 2
a = cnn_encoder(args)
x = np.ones((5,10))
x[0,1] = .5
x = Variable(torch.from_numpy(x.astype('int')))
a.forward(x)