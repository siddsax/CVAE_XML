import sys
import torch
sys.path.append('utils/')
sys.path.append('models/')
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
# from tensorflow.examples.tutorials.mnist import input_data
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

# x_tr = np.load('datasets/Eurlex/ft_trn.npy')
x_te = np.load('datasets/Eurlex/ft_tst.npy')
# y_tr = np.load('datasets/Eurlex/label_trn.npy')
y_te = np.load('datasets/Eurlex/label_tst.npy')

x_tr = np.load('datasets/Eurlex/eurlex_docs/x_tr.npy')
y_tr = np.load('datasets/Eurlex/eurlex_docs/y_tr.npy')

# -----------------------------

# -----------------------------

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



# - ------ legacy sampling ----------------------------------------------
# for i in range(np.shape(y_tr)[0]):
#     labels = ksort[:np.sum(k<3)]
#     fin_labels = np.random.choice(labels, x, replace=False)
#     new_y[i, fin_labels] = 1
#     # print(pcaed)
# ---------------------------------------------------------------------

# ---------- Cluster Sampling ---------------------------------------
# label_counts = np.sum(y_tr, axis=0)
# x = np.sum(y_tr, axis=1)
# new_y = np.zeros(np.shape(y_tr))#[0], np.shape(y_tr)[1])
# clusters = np.load('cluster_assignments_1.npy')[0,:]
# num_clusters = np.max(clusters)

# lives = label_counts - 40
# lives[np.argwhere(lives>0)] = 0
# clusters[np.argwhere(lives==0)] = num_clusters + 1 
# data_pts_num = []
# data_pts = []
# for i in range(num_clusters):
#     data_pts.append(np.argwhere(clusters==i))           
#     data_pts_num.append(len(data_pts[i]))


# data = 0
# priority_list = []
# stuck_count = 0
# while(np.sum(lives) < 0 and data < y_tr.shape[0]):
#     if(len(priority_list)):
#         clst_num = priority_list[0]
#         priority_list.remove(clst_num)
#     else:    
#         clst_num = np.random.randint(0, high=num_clusters)
#     num_labels = np.random.choice(x)
#     if(num_labels>data_pts_num[clst_num]):
#         if(stuck_count>10):
#             stuck_count = 0
#         else:
#             stuck_count+=1
#             priority_list.append(clst_num)
#             print(" ---- stuck ---- at {1} for {0} ----".format(num_labels, clst_num))
#             continue
#     else:
#         x = np.delete(x, np.argwhere(x==num_labels)[0])
#         fin_labels = np.random.choice(data_pts[clst_num][:,0], int(num_labels), replace=False)
#         lives[fin_labels] += 1
#         clusters[np.argwhere(lives==0)] = num_clusters + 1
#         data_pts_num = []
#         data_pts = []
#         for i in range(num_clusters):
#             data_pts.append(np.argwhere(clusters==i))           
#             data_pts_num.append(len(data_pts[i]))

#         new_y[data, fin_labels] = 1
#         data+=1
#         print(data)
# ---------------------------------------------------------------------

new_y = np.load('new_y.npy')
c = Variable(torch.from_numpy(new_y.astype('float32'))).type(dtype)
X_dim = x_tr.shape[1]
y_dim = y_tr.shape[1]
print(X_dim)
print(y_dim)
print(new_y.shape)
P = decoder(X_dim, y_dim, args.h_dim, args.Z_dim)
P.load_state_dict(torch.load('saved_models/' + args.model_name + '/P_best'))
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

if not os.path.exists('datasets/' + args.model_name ):
    os.makedirs('datasets/' + args.model_name)
np.save('datasets/' + args.model_name + '/new_x', np.around(new_x, decimals=4))
np.save('datasets/' + args.model_name + '/new_y', new_y)

