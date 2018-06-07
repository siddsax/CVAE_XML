import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
# from tensorflow.examples.tutorials.mnist import input_data
import sys
from sklearn import preprocessing
from sklearn.decomposition import PCA
import argparse
from encoder import encoder
from decoder import decoder


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
parser.add_argument('--mn', dest='model_name', type=str, default='default', help='model name')

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


x_tr = np.load('datasets/Eurlex/ft_trn.npy')
x_te = np.load('datasets/Eurlex/ft_tst.npy')
y_tr = np.load('datasets/Eurlex/label_trn.npy')
y_te = np.load('datasets/Eurlex/label_tst.npy')
if(args.pca_flag):
    pca = PCA(n_components=2)
    pca.fit(x_tr)
    x_tr = pca.transform(x_tr)
    x_te = pca.transform(x_te)
    pca = PCA(n_components=2)
    pca.fit(y_tr)
    y_tr = pca.transform(y_tr)
    y_te = pca.transform(y_te)

if(args.pp_flg):
    pp = preprocessing.MinMaxScaler()
else:
    pp = preprocessing.StandardScaler()

scaler = pp.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_te = scaler.transform(x_te)





k = np.sum(y_tr, axis=0)
ksort = k.argsort()
num_lbls = np.shape(k)[0]
k.sort()

x = int(np.ceil(np.mean(np.sum(y_tr, axis=1))))

new_y = np.zeros(np.shape(y_tr))#[0], np.shape(y_tr)[1])
for i in range(np.shape(y_tr)[0]):
    labels = ksort[:np.sum(k<2)]
    fin_labels = np.random.choice(labels, x, replace=False)
    new_y[i, fin_labels] = 1
    # print(pcaed)


c = Variable(torch.from_numpy(new_y.astype('float32'))).type(dtype)
X_dim = x_tr.shape[1]
y_dim = y_tr.shape[1]
P = decoder(X_dim, y_dim, args.h_dim, args.Z_dim)
P.load_state_dict(torch.load('saved_model/P_best'))
if(torch.cuda.is_available()):
    P.cuda()
    # Q.cuda()
    print("--------------- Using GPU! ---------")
else:
    print("=============== Using CPU =========")
############## TESTING ON!!! ###########
P.eval()
#########################################
eps = Variable(torch.randn(np.shape(y_tr)[0], args.Z_dim)).type(dtype)
inp = torch.cat([eps, c], 1).type(dtype)
X_sample = P.forward(inp)


new_x = scaler.inverse_transform(X_sample.data)
np.save('datasets/new_x', np.around(new_x, decimals=4))
np.save('datasets/new_y', new_y)



# import torch
# import torch.nn as nn
# import torch.autograd as autograd
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import os
# from torch.autograd import Variable
# # from tensorflow.examples.tutorials.mnist import input_data
# import sys
# from sklearn import preprocessing
# from sklearn.decomposition import PCA
# import argparse
# from encoder import encoder
# from decoder import decoder


# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
# parser.add_argument('--zd', dest='Z_dim', type=int, default=500, help='Latent layer dimension')
# parser.add_argument('--mb', dest='mb_size', type=int, default=10, help='Size of minibatch, changing might result in latent layer variance overflow')
# parser.add_argument('--hd', dest='h_dim', type=int, default=750, help='hidden layer dimension')
# parser.add_argument('--lr', dest='lr', type=int, default=1e-3, help='Learning Rate')
# parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
# parser.add_argument('--pp', dest='pp_flg', type=int, default=1, help='1 is for min-max pp, 0 is for gaussian pp')
# parser.add_argument('--s', dest='step', type=int, default=100, help='step for displaying loss')
# parser.add_argument('--b', dest='beta', type=int, default=1, help='Regularization param')
# parser.add_argument('--d', dest='disp_flg', type=int, default=1, help='display graphs')
# parser.add_argument('--sve', dest='save', type=int, default=0, help='save models or not')
# parser.add_argument('--ss', dest='save_step', type=int, default=5000, help='gap between model saves')
# parser.add_argument('--mn', dest='model_name', type=str, default='default', help='model name')

# args = parser.parse_args()

# if torch.cuda.is_available():
#     dtype = torch.cuda.FloatTensor
# else:
#     dtype = torch.FloatTensor


# x_tr = np.load('datasets/Eurlex/ft_trn.npy')
# x_te = np.load('datasets/Eurlex/ft_tst.npy')
# y_tr = np.load('datasets/Eurlex/label_trn.npy')
# y_te = np.load('datasets/Eurlex/label_tst.npy')
# if(args.pca_flag):
#     pca = PCA(n_components=2)
#     pca.fit(x_tr)
#     x_tr = pca.transform(x_tr)
#     x_te = pca.transform(x_te)
#     pca = PCA(n_components=2)
#     pca.fit(y_tr)
#     y_tr = pca.transform(y_tr)
#     y_te = pca.transform(y_te)

# if(args.pp_flg):
#     pp = preprocessing.MinMaxScaler()
# else:
#     pp = preprocessing.StandardScaler()

# scaler = pp.fit(x_tr)
# x_tr = scaler.transform(x_tr)
# x_te = scaler.transform(x_te)





# k = np.sum(y_tr, axis=0)
# ksort = k.argsort()
# num_lbls = np.shape(k)[0]
# k.sort()



# num_train = np.shape(y_tr)[0]
# x = np.random.normal(np.mean(np.sum(y_tr, axis=1)), np.var(np.sum(y_tr, axis=1)), num_train)
# x_cl_or_fl = np.random.binomial(1, .5, num_train)

# new_y = np.zeros(np.shape(y_tr))
# for i in range(num_train):
#     labels = ksort[:np.sum(k<2)]
#     if(x_cl_or_fl[i]):
#         x[i] = int(x[i])
#     else:
#         x[i] = np.ceil(x[i])
#     fin_labels = np.random.choice(labels, int(x[i]), replace=False)
#     new_y[i, fin_labels] = 1
#     # print(pcaed)


# c = Variable(torch.from_numpy(new_y.astype('float32'))).type(dtype)
# X_dim = x_tr.shape[1]
# y_dim = y_tr.shape[1]
# P = decoder(X_dim, y_dim, args.h_dim, args.Z_dim)
# P.load_state_dict(torch.load('saved_model/P_best'))
# if(torch.cuda.is_available()):
#     P.cuda()
#     # Q.cuda()
#     print("--------------- Using GPU! ---------")
# else:
#     print("=============== Using CPU =========")
# ############## TESTING ON!!! ###########
# P.eval()
# #########################################
# eps = Variable(torch.randn(num_train, args.Z_dim)).type(dtype)
# inp = torch.cat([eps, c], 1).type(dtype)
# X_sample = P.forward(inp)


# new_x = scaler.inverse_transform(X_sample.data)
# np.save('datasets/new_x', np.around(new_x, decimals=4))
# np.save('datasets/new_y', new_y)