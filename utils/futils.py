import os
import sys
import torch
import timeit
import argparse
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec



def count_parameters(model):
    a = 0
    for p in model.parameters():
        if p.requires_grad:
            a += p.numel()
            # print("Num params of the paramter filled layer {0}".format(p.numel())) #, Num params {1}
        # else:
            # print("This layer is without params")# {0}".format(type(p).__name__,))
    # print("*"*100)
    return a

def effective_k(k, d):
    return (k - 1) * d + 1

def sample_z(mu, log_var, params, dtype_f):
    eps = Variable(torch.randn(params.batch_size, params.Z_dim).type(dtype_f))
    return mu + torch.exp(log_var / 2) * eps

def gen_model_file(params):
    data_name = params.data_path.split('/')[-2]
    fs_string = '-'.join([str(fs) for fs in params.filter_sizes])
    file_name = 'data-%s_sl-%d_ed-%d_fs-%s_nf-%d_pu-%d_pt-%s_hd-%d_bs-%d_model-%s_pretrain-%s' % \
        (data_name, params.sequence_length, params.embedding_dim,
         fs_string, params.num_filters, params.pooling_units,
         params.pooling_type, params.hidden_dims, params.batch_size,
         params.model_variation, params.pretrain_type)
    return file_name
