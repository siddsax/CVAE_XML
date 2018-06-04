import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import sys
from sklearn import preprocessing
from sklearn.decomposition import PCA

from encoder import encoder
from decoder import decoder

x_tr = np.load('datasets/Eurlex/ft_trn.npy')
pca = PCA(n_components=2)
pca.fit(x_tr)
x_te = np.load('datasets/Eurlex/ft_tst.npy')
x_tr = pca.transform(x_tr)
x_te = pca.transform(x_te)
y_tr = np.load('datasets/Eurlex/label_trn.npy')
y_te = np.load('datasets/Eurlex/label_tst.npy')
pca = PCA(n_components=2)
pca.fit(y_tr)
y_tr = pca.transform(y_tr)
y_te = pca.transform(y_te)
scaler = preprocessing.StandardScaler().fit(x_tr)
x_tr = scaler.transform(x_tr)
x_te = scaler.transform(x_te)
scaler = preprocessing.StandardScaler().fit(y_tr)
y_tr = scaler.transform(y_tr)
y_te = scaler.transform(y_te)


mb_size = 10
Z_dim = int(sys.argv[1])
X_dim = x_tr.shape[1]#mnist.train.images.shape[1]
y_dim = y_tr.shape[1]#mnist.train.labels.shape[1]
N = x_tr.shape[0]
h_dim = 128
cnt = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         xavier(m.weight.data)
#         xavier(m.bias.data)

# =============================== Q(z|X) ======================================

Q = encoder(X_dim, y_dim, h_dim, Z_dim)

def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P = decoder(X_dim, y_dim, h_dim, Z_dim)

# =============================== TRAINING ====================================

optimizer = optim.Adam(list(P.parameters()) + list(Q.parameters()), lr=lr)
loss_fn = torch.nn.BCEWithLogitsLoss(size_average=False)
loss_best = 1000
for it in range(10000000):
    a = it*mb_size%(N-mb_size)
    X, c = x_tr[a:a+mb_size], y_tr[a:a+mb_size]
    X = Variable(torch.from_numpy(X.astype('float32')))
    c = Variable(torch.from_numpy(c.astype('float32')))

    # Forward
    inp = torch.cat([X, c],1)
    z_mu, z_var  = Q.forward(inp)
    z = sample_z(z_mu, z_var)
    inp = torch.cat([z, c], 1)
    X_sample = P.forward(inp)

    # Loss
    recon_loss = loss_fn(X_sample, X) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss

    if(recon_loss<0):
        print(recon_loss)
        print(X_sample[0:100])
        print(X[0:100])
        sys.exit()
    # print(kl_loss)
    # print(torch.max(z_var))

    # Backward
    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print and plot every now and then
    if it % 10000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data))
        if(int(sys.argv[2])):

            fig = plt.figure(figsize=(4, 4))
            z = z.data.numpy()
            plt.scatter(z[:,0], z[:,1],c=act_test.data.numpy())

            if not os.path.exists('out/'):
                os.makedirs('out/')

            plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        
        if not os.path.exists('saved_model/'):
            os.makedirs('saved_model/')
        torch.save(P.state_dict(), "saved_model/P_" + str(cnt))
        torch.save(Q.state_dict(), "saved_model/Q_"+ str(cnt))

        if(loss<loss_best):
            loss_best = loss
            torch.save(P.state_dict(), "saved_model/P_best")
            torch.save(Q.state_dict(), "saved_model/Q_best")

