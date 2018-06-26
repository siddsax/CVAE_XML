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
import data_helpers
import scipy
import subprocess
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

def load_data(params):
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, params = data_helpers.load_data(params, max_length=params.sequence_length, vocab_size=params.vocab_size)
    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)
    return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, params


def sample_z(mu, log_var, params):
    eps = Variable(torch.randn(log_var.shape[0], params.Z_dim).type(params.dtype))
    k = torch.exp(log_var / 2) * eps
    return mu + k

def load_data(X, Y, params, batch=True):
    if(batch):
        a = np.random.randint(0,params.N, size=params.mb_size)
        if isinstance(X, scipy.sparse.csr.csr_matrix) or isinstance(X, scipy.sparse.csc.csc_matrix):
            X, c = X[a].todense(), Y[a].todense()
        else:
            X, c = X[a], Y[a]
            
    else:
        if isinstance(X, scipy.sparse.csr.csr_matrix) or isinstance(X, scipy.sparse.csc.csc_matrix):
            X, c = X.todense(), Y.todense()
        else:
            X, c = X, Y
    
    X = Variable(torch.from_numpy(X.astype('float32')).type(params.dtype))
    Y = Variable(torch.from_numpy(c.astype('float32')).type(params.dtype))
    return X,Y

def write_grads(model, thefile):
    grads = []
    for key, value in model.named_parameters():
        if(value.grad is not None):
            grads.append(value.grad.mean().squeeze().cpu().numpy())

    thefile = open('gradient_classifier.txt', 'a+')
    for item in grads:
        thefile.write("%s " % item)
    thefile.write("\n" % item)
    thefile.close()