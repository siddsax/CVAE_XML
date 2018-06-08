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


# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

loss_func = torch.nn.BCEWithLogitsLoss()

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



print(x_tr[0:100,:])
print(y_tr[0:100,:])


mb_size = 15000
Z_dim = int(sys.argv[1])
X_dim = x_tr.shape[1]#mnist.train.images.shape[1]
y_dim = y_tr.shape[1]#mnist.train.labels.shape[1]
N = x_tr.shape[0]
h_dim = 128
cnt = 0
lr = 1e-3

x_test = x_te
x_test = Variable(torch.from_numpy(x_test))
y_test = y_te
y_test = c = Variable(torch.from_numpy(y_test.astype('float32')))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

# =============================== P(z|X) ======================================

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def CP(X):
    inputs = X#torch.cat([X, c], 1)
    h = nn.relu(torch.matmul(inputs , Wxh) + bxh.repeat(inputs.size(0), 1))
    z_mu = torch.matmul(h , Whz_mu) + bhz_mu.repeat(h.size(0), 1)
    z_var = torch.matmul(h , Whz_var) + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mu.shape[0], Z_dim))
    return mu + torch.exp(log_var / 2) * eps



# =============================== Q(z|X, y) ======================================

Wxyh = xavier_init(size=[X_dim + y_dim, h_dim])
bxyh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz2_mu = xavier_init(size=[h_dim, Z_dim])
bhz2_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz2_var = xavier_init(size=[h_dim, Z_dim])
bhz2_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def Rec(X, c):
    inputs = torch.cat([X, c], 1)
    h = nn.relu(torch.matmul(inputs , Wxyh) + bxyh.repeat(inputs.size(0), 1))
    z_mu = torch.matmul(h , Whz2_mu) + bhz2_mu.repeat(h.size(0), 1)
    z_var = torch.matmul(h , Whz2_var) + bhz2_var.repeat(h.size(0), 1)
    return z_mu, z_var

# =============================== P(y|z, x) ======================================

Wzh = xavier_init(size=[Z_dim + X_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, y_dim])
bhx = Variable(torch.zeros(y_dim), requires_grad=True)


def Gen(z, X):
    inputs = torch.cat([z, X], 1)
    h = nn.relu(torch.matmul(inputs , Wzh) + bzh.repeat(inputs.size(0), 1))
    y = nn.relu(torch.matmul(h , Whx) + bhx.repeat(h.size(0), 1))
    return y

def Gen_test(z, X):
    inputs = torch.cat([z, X], 1)
    h = nn.relu(torch.matmul(inputs , Wzh) + bzh.repeat(inputs.size(0), 1))
    y = nn.sigmoid(torch.matmul(h , Whx) + bhx.repeat(h.size(0), 1))
    return y

# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, 
          Wxyh, bxyh, Whz2_mu, bhz2_mu, Whz2_var, bhz2_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)

for it in range(100000):
    a = it*mb_size%(N-mb_size)
    X, c = x_tr[a:a+mb_size], y_tr[a:a+mb_size]
    X = Variable(torch.from_numpy(X.astype('float32')))
    c = Variable(torch.from_numpy(c.astype('float32')))

    # Forward
    z_mu1, lg_z_var1 = Rec(X, c)
    z_mu2, lg_z_var2 = CP(X)
    z1 = sample_z(z_mu1, lg_z_var1)
    z2 = sample_z(z_mu2, lg_z_var2)
    y1_sample = Gen(z1, X)
    y2_sample = Gen(z2, X)

    # Loss
    # recon_loss_1 = loss(y1_sample, c, size_average=False) / mb_size
    # recon_loss_2 = loss(y2_sample, c, size_average=False) / mb_size
    recon_loss_1 = loss_func(y1_sample, c) / mb_size
    recon_loss_2 = loss_func(y2_sample, c) / mb_size

    # kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    kl_loss = torch.mean(0.5 * torch.sum(  (lg_z_var2-lg_z_var1) + (torch.exp(lg_z_var1) + (z_mu1 - z_mu2)**2)/torch.exp(lg_z_var2) - 1., 1   ))
    loss = recon_loss_1 + recon_loss_2 + kl_loss

    print(kl_loss)
    # sys.exit()
    # print("=======")
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
    if it % 10 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data))
        # pred = torch.argmax(y2_sample, 1)
        # act = torch.argmax(c, 1)
        
        # zm, lg_zv = CP(x_test)
        # # print(zm.shape)
        # # print(lg_zv.shape)
        # z = sample_z(zm, lg_zv)
        
        # k = Gen(z, x_test)
        # # print(k)
        # y_pred_test = torch.argmax(k,1)
        # act_test = torch.argmax(y_test, 1)

        # classification_acc = torch.mean((pred==act).type(torch.float))
        # classification_acc_test = torch.mean((y_pred_test==act_test).type(torch.float))
        # print('Iter-{}; Liklihood: {:.4}; train Acc: {:.4}; test Acc: {:.4}'.format(it, -loss.data[0], 100*classification_acc, 100*classification_acc_test))

        if(int(sys.argv[2])):
            # c = np.zeros(shape=[mb_size, y_dim], dtype='float32')
            # c[:, np.random.randint(0, 10)] = 1.
            # c = Variable(torch.from_numpy(c))
            # z = Variable(torch.randn(mb_size, Z_dim))
            # samples = P(z, c).data.numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            z = z.data.numpy()
            plt.scatter(z[:,0], z[:,1],c=act_test.data.numpy())
            # gs = gridspec.GridSpec(4, 4)
            # gs.update(wspace=0.05, hspace=0.05)
            
            # for i in range(z.data.shape[0]):
            #     plt.plot(z.data[i], c=act_test.data[i])
            # plt.show()
            # for i, sample in enumerate(samples):
            #     ax = plt.subplot(gs[i])
            #     plt.axis('off')
            #     ax.set_xticklabels([])
            #     ax.set_yticklabels([])
            #     ax.set_aspect('equal')
            #     plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('out/'):
                os.makedirs('out/')

            plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
            cnt += 1
            # plt.close(fig)


