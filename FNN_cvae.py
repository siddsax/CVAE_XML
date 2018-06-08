import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
#from tensorflow.examples.tutorials.mnist import input_data
import sys
from sklearn import preprocessing
from sklearn.decomposition import PCA

from encoder import encoder
from decoder import decoder
import argparse
from visdom import Visdom

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
parser.add_argument('--mn', dest='model_name', type=str, default='default', help='model name')

args = parser.parse_args()


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

# =============================== Q(z|X) ======================================
Q = encoder(X_dim, y_dim, args.h_dim, args.Z_dim)

def sample_z(mu, log_var):
    eps = Variable(torch.randn(args.mb_size, args.Z_dim).type(dtype))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P = decoder(X_dim, y_dim, args.h_dim, args.Z_dim)

# =============================== TRAINING ====================================
if(torch.cuda.is_available()):
    P.cuda()
    Q.cuda()
    print("--------------- Using GPU! ---------")
else:
    print("=============== Using CPU =========")
    
optimizer = optim.Adam(list(P.parameters()) + list(Q.parameters()), lr=args.lr)



loss_best = 1000.0
loss_best2 = 1000.0
for it in range(1000000):
    a = it*args.mb_size%(N-args.mb_size)
    X, c = x_tr[a:a+args.mb_size], y_tr[a:a+args.mb_size]
    X = Variable(torch.from_numpy(X.astype('float32')).type(dtype))
    c = Variable(torch.from_numpy(c.astype('float32')).type(dtype))

        
    # Forward
    inp = torch.cat([X, c],1).type(dtype)
    z_mu, z_var  = Q.forward(inp)
    z = sample_z(z_mu, z_var)
    inp = torch.cat([z, c], 1).type(dtype)
    X_sample = P.forward(inp)

    # Loss
    # recon_loss = loss_fn(X_sample, X) / args.mb_size
    recon_loss = torch.mean(torch.norm(X_sample - X))# / args.mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + args.beta*kl_loss

    if(recon_loss<0):
        print(recon_loss)
        print(X_sample[0:100])
        print(X[0:100])
        sys.exit()

    # Backward
    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print and plot every now and then
    if it % args.step == 0:
        if(args.disp_flg):
            if(it==0):
                loss_old = loss.data
            else:
                viz.line(X=np.linspace(it-args.step,it,50), Y=np.linspace(loss_old, loss.data,50), name='1', update='append', win=win)
                loss_old = loss.data
            if(it % 1500 == 0 ):
                win = viz.line(X=np.arange(it, it + .1), Y=np.arange(0, .1))
        print('Iter-{}; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(it, loss.data, kl_loss.data, recon_loss.data, loss_best2))
        
        if(loss<loss_best2):
            loss_best2 = loss

        if(args.plot_flg):
            fig = plt.figure(figsize=(4, 4))
            z = z.data.numpy()
            plt.scatter(z[:,0], z[:,1],c=act_test.data.numpy())
            if not os.path.exists('out/'):
                os.makedirs('out/')
            plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')

    if args.save:
        
        if it % args.save_step == 0:
            if not os.path.exists('saved_models/' + args.model_name ):
                os.makedirs('saved_models/' + args.model_name)
            torch.save(P.state_dict(), "saved_models/" + args.model_name + "/P_" + str(it))
            torch.save(Q.state_dict(), "saved_models/" + args.model_name + "/Q_"+ str(it))
            cnt += 1

        if(loss<loss_best):
            loss_best = loss
            torch.save(P.state_dict(), "saved_models/" + args.model_name + "/P_best")
            torch.save(Q.state_dict(), "saved_models/" + args.model_name + "/Q_best")

