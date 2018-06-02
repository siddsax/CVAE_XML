import torch
import torch.nn.functional as nn
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


# =============================== Q(z|X) ======================================

Wxh = xavier_init(size=[X_dim + y_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def Q(X, c):
    inputs = torch.cat([X, c], 1)
    h = nn.relu(torch.matmul(inputs , Wxh) + bxh.repeat(inputs.size(0), 1))
    z_mu = torch.matmul(h , Whz_mu) + bhz_mu.repeat(h.size(0), 1)
    z_var = torch.matmul(h , Whz_var) + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

Wzh = xavier_init(size=[Z_dim + y_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def P(z, c):
    inputs = torch.cat([z, c], 1)
    h = nn.relu(torch.matmul(inputs , Wzh) + bzh.repeat(inputs.size(0), 1))
    X = nn.sigmoid(torch.matmul(h , Whx) + bhx.repeat(h.size(0), 1))
    return X


# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)

for it in range(100000):
    a = it*mb_size%(N-mb_size)
    X, c = x_tr[a:a+mb_size], y_tr[a:a+mb_size]
    X = Variable(torch.from_numpy(X.astype('float32')))
    c = Variable(torch.from_numpy(c.astype('float32')))

    # Forward
    z_mu, z_var = Q(X, c)
    z = sample_z(z_mu, z_var)
    X_sample = P(z, c)

    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss

    # print(kl_loss)
    # print(torch.max(z_var))

    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data))
        if(int(sys.argv[2])):

            fig = plt.figure(figsize=(4, 4))
            z = z.data.numpy()
            plt.scatter(z[:,0], z[:,1],c=act_test.data.numpy())

            if not os.path.exists('out/'):
                os.makedirs('out/')

            plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
            cnt += 1
        
        # if not os.path.exists('saved_model/'):
        #     os.makedirs('saved_model/')
        # torch.save(the_model.state_dict(), "saved_model/")
