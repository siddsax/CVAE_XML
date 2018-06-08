import os
import sys
import torch
import timeit
import argparse
import numpy as np
import data_helpers
import torch.nn as nn
from visdom import Visdom
from encoder import encoder
from decoder import decoder
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec



viz = Visdom()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
parser.add_argument('--zd', dest='Z_dim', type=int, default=500, help='Latent layer dimension')
parser.add_argument('--mb', dest='mb_size', type=int, default=10, help='Size of minibatch, changing might result in latent layer variance overflow')
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

parser.add_argument('--data_path',help='raw data path in CPickle format', type=str, default='../sample_data/rcv1_raw_small.p')
parser.add_argument('--sequence_length',help='max sequence length of a document', type=int,default=500)
parser.add_argument('--embedding_dim', help='dimension of word embedding representation', type=int, default=300)
parser.add_argument('--filter_sizes', help='number of filter sizes (could be a list of integer)', type=int, default=[2, 4, 8], nargs='+')
parser.add_argument('--num_filters', help='number of filters (i.e. kernels) in CNN model', type=int, default=32)
parser.add_argument('--pooling_units', help='number of pooling units in 1D pooling layer', type=int, default=32)
parser.add_argument('--pooling_type', help='max or average', type=str, default='max')
parser.add_argument('--hidden_dims', help='number of hidden units', type=int, default=512)
parser.add_argument('--model_variation', help='model variation: CNN-rand or CNN-pretrain', type=str, default='pretrain')
parser.add_argument('--pretrain_type', help='pretrain model: GoogleNews or glove', type=str, default='glove')
parser.add_argument('--batch_size', help='number of batch size', type=int, default=256)
parser.add_argument('--num_epochs', help='number of epcohs for training', type=int, default=50)
parser.add_argument('--vocab_size', help='size of vocabulary keeping the most frequent words', type=int, default=30000)
args = parser.parse_args()

if(len(args.model_name)==0):
    args.model_name = gen_model_file(args)

def load_data(args):
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv = data_helpers.load_data(args.data_path, max_length=args.sequence_length, vocab_size=args.vocab_size)
    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)
    return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv

def gen_model_file(args):
    data_name = args.data_path.split('/')[-2]
    fs_string = '-'.join([str(fs) for fs in args.filter_sizes])
    file_name = 'data-%s_sl-%d_ed-%d_fs-%s_nf-%d_pu-%d_pt-%s_hd-%d_bs-%d_model-%s_pretrain-%s' % \
        (data_name, args.sequence_length, args.embedding_dim,
         fs_string, args.num_filters, args.pooling_units,
         args.pooling_type, args.hidden_dims, args.batch_size,
         args.model_variation, args.pretrain_type)
    return file_name

print('-'*50)
print('Loading data...'); start_time = timeit.default_timer();
x_tr, y_tr, x_te, y_te, vocabulary, vocabulary_inv = load_data(args)
print('Process time %.3f (secs)\n' % (timeit.default_timer() - start_time))


if(args.pp_flg):
    pp = preprocessing.MinMaxScaler()
else:
    pp = preprocessing.StandardScaler()

loss_fn = torch.nn.BCELoss(size_average=False)
# loss_fn = torch.nn.MSELoss(size_average=False)

scaler = pp.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_te = scaler.transform(x_te)

X_dim = x_tr.shape[1]
y_dim = y_tr.shape[1]
N = x_tr.shape[0]
cnt = 0


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


print 'model_variaton:', args.model_variation
if args.model_variation=='pretrain':
    embedding_weights = load_word2vec(args.pretrain_type, vocabulary_inv, args.embedding_dim)
elif args.model_variation=='random':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')
embedding_weights = embedding_weights
vocab_size = len(vocabulary)


