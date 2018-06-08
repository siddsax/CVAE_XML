import sys
import scipy
from scipy.sparse import *
from scipy.io import savemat
import numpy as np
import os

x = np.load('datasets/' + sys.argv[1] + '/new_x.npy')
y = np.load('datasets/' + sys.argv[1] + '/new_y.npy')
x_o = np.load('datasets/Eurlex/ft_trn.npy')
y_o = np.load('datasets/Eurlex/label_trn.npy')
x_n = np.concatenate((x,x_o),axis=0)
y_n = np.concatenate((y,y_o),axis=0)

x_n_sp = csr_matrix(x_n)
y_n_sp = csr_matrix(y_n)

if not os.path.exists('../fastxml/manik/train_datasets/' + sys.argv[1] ):
    os.makedirs('../fastxml/manik/train_datasets/' + sys.argv[1])
savemat('../fastxml/manik/train_datasets/' + sys.argv[1] + '/x_big', {'x_big':x_n_sp})
savemat('../fastxml/manik/train_datasets/' + sys.argv[1] + '/y_big', {'y_big':y_n_sp})