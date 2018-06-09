import numpy as np

x = np.load('datasets/new_x.npy')
y = np.load('datasets/new_y.npy')
x_o = np.load('datasets/Eurlex/ft_trn.npy')
y_o = np.load('datasets/Eurlex/label_trn.npy')
x_n = np.concatenate((x,x_o),axis=0)
y_n = np.concatenate((y,y_o),axis=0)

import scipy
from scipy.sparse import *
x_n_sp = csr_matrix(x_n)
y_n_sp = csr_matrix(y_n)
from scipy.io import savemat
savemat('../fastxml/manik/x_big', {'x_big':x_n_sp})
savemat('../fastxml/manik/y_big', {'y_big':y_n_sp})