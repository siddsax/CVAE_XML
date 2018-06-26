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
sys.path.append('utils/')
sys.path.append('models/')
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
from encoder import encoder
from decoder import decoder
import argparse
from visdom import Visdom
from sklearn.externals import joblib 
from futils import *
from loss import loss
import time

viz = Visdom()
# ------------------------ Params -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
parser.add_argument('--zd', dest='Z_dim', type=int, default=200, help='Latent layer dimension')
parser.add_argument('--mb', dest='mb_size', type=int, default=100, help='Size of minibatch, changing might result in latent layer variance overflow')
parser.add_argument('--hd', dest='h_dim', type=int, default=600, help='hidden layer dimension')
parser.add_argument('--lr', dest='lr', type=int, default=1e-4, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
parser.add_argument('--e', dest='num_epochs', type=int, default=100, help='step for displaying loss')
parser.add_argument('--b', dest='beta', type=float, default=1.0, help='factor multipied to likelihood param')
parser.add_argument('--d', dest='disp_flg', type=int, default=0, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=1, help='save models or not')
parser.add_argument('--ss', dest='save_step', type=int, default=100, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
parser.add_argument('--tr', dest='training', type=int, default=1, help='model name')
parser.add_argument('--lm', dest='load_model', type=str, default="", help='model name')
parser.add_argument('--loss', dest='loss_type', type=str, default="L1Loss", help='model name')
parser.add_argument('--fl', dest='fin_layer', type=str, default="ReLU", help='model name')
parser.add_argument('--pp', dest='pp_flg', type=int, default=1, help='1 is for min-max pp, 2 is for gaussian pp, 0 for none')

args = parser.parse_args()

x_tr = np.load('datasets/Eurlex/eurlex_docs/x_tr.npy')
y_tr = np.load('datasets/Eurlex/eurlex_docs/y_tr.npy')

if(args.pp_flg):
    if(args.pp_flg==1):
        pp = preprocessing.MinMaxScaler()
    elif(args.pp_flg==2):
        pp = preprocessing.StandardScaler()
    scaler = pp.fit(x_tr)
    x_tr = scaler.transform(x_tr)

X_dim = x_tr.shape[1]
y_dim = y_tr.shape[1]
N = x_tr.shape[0]

def sample_z(mu, log_var):
    eps = Variable(torch.randn(log_var.shape[0], args.Z_dim).type(dtype))
    k = torch.exp(log_var / 2) * eps
    return mu + k

P = torch.load(args.load_model + "/P_best", map_location=lambda storage, loc: storage)
Q = torch.load(args.load_model + "/Q_best", map_location=lambda storage, loc: storage)
P_weights = [P.l0.data, P.l2.data] 
Q_weights = [P.l0.data, P.mu.data, P.var.data]

print(P_weights)
