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


Z_dim = int(sys.argv[1])
h_dim = 128
# x_te = np.load('datasets/Eurlex/ft_tst.npy')
# x_tr = pca.transform(x_tr)
# x_te = pca.transform(x_te)
y_tr = np.load('datasets/Eurlex/label_trn.npy')
y_te = np.load('datasets/Eurlex/label_tst.npy')

pca = PCA(n_components=2)
pca.fit(y_tr)
y_tr2 = pca.transform(y_tr)
y_te2 = pca.transform(y_te)
# x_tr = scaler.transform(x_tr)
# x_te = scaler.transform(x_te)
scaler = preprocessing.StandardScaler().fit(y_tr2)
# y_tr2 = scaler.transform(y_tr2)
# y_te2 = scaler.transform(y_te2)


k = np.sum(y_tr, axis=0)
ksort = k.argsort()
num_lbls = np.shape(k)[0]
k.sort()


x = int(np.mean(np.sum(y_tr, axis=1)))

# rnge = np.array(range(num_lbls))
# print(np.shape(rnge))
# print(np.shape(k))
# plt.bar(rnge, k, align='center')
# plt.show()
# print(k)
# print(num_lbls)

new_y = np.zeros(np.shape(y_tr))#[0], np.shape(y_tr)[1])
for i in range(np.shape(y_tr)[0]):
    labels = ksort[:np.sum(k<2)]
    fin_labels = np.random.choice(labels, x, replace=False)
    new_y[i, fin_labels] = 1
    # print(pcaed)

new_y = pca.transform(new_y)
new_y = scaler.transform(new_y)

c = Variable(torch.from_numpy(new_y.astype('float32')))
X_dim = 2
y_dim = 2
P = decoder(X_dim, y_dim, h_dim, Z_dim)
P.load_state_dict(torch.load('saved_model/P_best'))

eps = Variable(torch.randn(np.shape(y_tr)[0], Z_dim))
inp = torch.cat([eps, c], 1)
X_sample = P.forward(inp)

x_tr = np.load('datasets/Eurlex/ft_trn.npy')
pca = PCA(n_components=2)
pca.fit(x_tr)
x_tr = pca.transform(x_tr)
scaler = preprocessing.StandardScaler().fit(x_tr)
new_x = pca.inverse_transform(scaler.inverse_transform(X_sample.data)) 
np.save('new_x', new_x)
np.save('new_y', new_y)