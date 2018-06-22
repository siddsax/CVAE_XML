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

def weights_init(m):
    torch.nn.init.xavier_uniform_(m.weight.data)

def get_gpu_memory_map(boom, name=False):
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    if(name):
        print("In " + str(name) + " Print: {0}; Mem(1): {1}; Mem(2): {2}; Mem(3): {3}; Mem(4): {4}".format( boom, gpu_memory_map[0], \
        gpu_memory_map[1], gpu_memory_map[2], gpu_memory_map[3]))
    else:
        print("Print: {0}; Mem(1): {1}; Mem(2): {2}; Mem(3): {3}; Mem(4): {4}".format( boom, gpu_memory_map[0], \
        gpu_memory_map[1], gpu_memory_map[2], gpu_memory_map[3]))
    return boom+1


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
    k = torch.exp(log_var / 2) * eps
    return mu + k
def gen_model_file(params):
    data_name = params.data_path.split('/')[-2]
    fs_string = '-'.join([str(fs) for fs in params.filter_sizes])
    file_name = 'data-%s_sl-%d_ed-%d_fs-%s_nf-%d_pu-%d_pt-%s_hd-%d_bs-%d_model-%s_pretrain-%s_beta-%s' % \
        (data_name, params.sequence_length, params.embedding_dim,
         fs_string, params.num_filters, params.pooling_units,
         params.pooling_type, params.hidden_dims, params.batch_size,
         params.model_variation, params.pretrain_type, params.beta)
    return file_name

def bce_loss(y_pred, y):
    y_pred_1 = torch.log(y_pred)
    y_pred_2 = torch.log(1 - y_pred)
    t = -torch.sum(torch.mean(y_pred_1*y + y_pred_2*(1-y),dim=0))
    if(t<0):
        print(y_pred)
        print(y_pred_1)
        print(y_pred*y)
        print(y_pred_1*(1-y))
        print(torch.mean(y_pred*y + y_pred_1*(1-y),dim=0))
        print(t)
    return t
