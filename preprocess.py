import os
import sys
import torch
import timeit
import argparse
import numpy as np
sys.path.append('utils/')
sys.path.append('models/')
import data_helpers 
from perplexity import Perplexity
import time
import torch.nn as nn
from w2v import *
from visdom import Visdom
from embedding_layer import embedding_layer
from cnn_decoder import cnn_decoder
from cnn_encoder import cnn_encoder
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from scipy import sparse 

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
parser.add_argument('--zd', dest='Z_dim', type=int, default=500, help='Latent layer dimension')
parser.add_argument('--hd', dest='h_dim', type=int, default=750, help='hidden layer dimension')
parser.add_argument('--lr', dest='lr', type=int, default=1e-3, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
parser.add_argument('--pp', dest='pp_flg', type=int, default=1, help='1 is for min-max pp, 0 is for gaussian pp')
parser.add_argument('--s', dest='step', type=int, default=100, help='step for displaying loss')
parser.add_argument('--b', dest='beta', type=int, default=1, help='Regularization param')
parser.add_argument('--d', dest='disp_flg', type=int, default=1, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=0, help='save models or not')
parser.add_argument('--ss', dest='save_step', type=int, default=5000, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')

parser.add_argument('--data_path',help='raw data path in CPickle format', type=str, default='datasets/rcv/rcv1_raw_small.p')
parser.add_argument('--sequence_length',help='max sequence length of a document', type=int,default=500)
parser.add_argument('--embedding_dim', help='dimension of word embedding representation', type=int, default=300)
parser.add_argument('--filter_sizes', help='number of filter sizes (could be a list of integer)', type=int, default=[2, 4, 8], nargs='+')
parser.add_argument('--num_filters', help='number of filters (i.e. kernels) in CNN model', type=int, default=32)
parser.add_argument('--pooling_units', help='number of pooling units in 1D pooling layer', type=int, default=32)
parser.add_argument('--pooling_type', help='max or average', type=str, default='max')
parser.add_argument('--hidden_dims', help='number of hidden units', type=int, default=512)
parser.add_argument('--model_variation', help='model variation: CNN-rand or CNN-pretrain', type=str, default='pretrain')
parser.add_argument('--pretrain_type', help='pretrain model: GoogleNews or glove', type=str, default='glove')
parser.add_argument('--batch_size', help='number of batch size', type=int, default=45)
parser.add_argument('--num_epochs', help='number of epcohs for training', type=int, default=50)
parser.add_argument('--vocab_size', help='size of vocabulary keeping the most frequent words', type=int, default=30000)
parser.add_argument('--training', help='training means 1, testing means 0', type=int, default=1)
parser.add_argument('--drop_prob', help='Dropout probability', type=int, default=.3)
parser.add_argument('--load_data', help='Load Data or not', type=int, default=1)


params = parser.parse_args()
params.pad_token = "<PAD/>"
params.go_token = '<GO/>'
params.end_token = '<END/>'

def load_data(params):
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, params = data_helpers.load_data(params, max_length=params.sequence_length, vocab_size=params.vocab_size)
    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)
    return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, params

data_path = "/".join(params.data_path.split('/')[:-1])
if(params.load_data):
    # print(load_data)
    print("Loading Data")
    print(data_path)
    x_tr, y_tr, x_te, y_te, vocabulary, vocabulary_inv, params = load_data(params)

    np.save(data_path + '/x_train', x_tr)
    sparse.save_npz(data_path + '/y_train', y_tr)
    sparse.save_npz(data_path + '/y_test', y_te)
    np.save(data_path + '/x_test', x_te)
    np.save(data_path + '/vocab', vocabulary)
    np.save(data_path + '/vocab_inv', vocabulary_inv)

np.save(data_path + '/params', params)