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

def effective_k(k, d):
    """
    :param k: kernel width
    :param d: dilation size
    :return: effective kernel width when dilation is performed
    """
    return (k - 1) * d + 1

def sample_z(mu, log_var):
    eps = Variable(torch.randn(args.mb_size, args.Z_dim).type(dtype))
    return mu + torch.exp(log_var / 2) * eps

parser = argparse.ArgumentParser(description='Process some integers.')
params = parser.parse_args()
params.a = 1
# Model Hyperparameters
params.sequence_length = 10
params.embedding_dim = 2
params.filter_sizes = [2, 4]
params.num_filters =  3
params.pooling_units = 2
params.pooling_type = 'max'
params.hidden_dims = 2
params.Z_dim = 2
# Training Hyperparameters
params.batch_size = 2
params.num_epochs = 2
# Model variation and w2v pretrain type
params.model_variation = 'as'
params.vocab_size = 2


params.decoder_kernels = [(400, params.Z_dim + params.embedding_dim, 3),
                                (450, 400, 3),
                                (500, 450, 3)]
params.decoder_dilations = [1, 2, 4]
params.decoder_paddings = [effective_k(w, self.decoder_dilations[i]) - 1
                                 for i, (_, _, w) in enumerate(params.decoder_kernels)]

if args.model_variation=='pretrain':
    embedding_weights = load_word2vec(args.pretrain_type, vocabulary_inv, args.embedding_dim)
else args.model_variation=='random':
    embedding_weights = None

en = embedding_layer(params, embedding_weights)
a = cnn_encoder(params)
x = np.ones((5,10))
x[0,1] = .5
x = Variable(torch.from_numpy(x.astype('int')))
e_emb = en.forward(x)
z_mu, z_lvar = a.forward(e_emb)
z = sample_z(z_mu, z_lvar)