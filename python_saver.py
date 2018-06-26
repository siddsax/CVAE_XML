import numpy as np

x = np.load('datasets/clustering_y/new_x.npy')
y = np.load('datasets/clustering_y/new_y.npy')
x_o = np.load('/scratch/work/saxenas2/CVAE_XML/datasets/Eurlex/eurlex_docs/x_tr.npy')
xt_o = np.load('/scratch/work/saxenas2/CVAE_XML/datasets/Eurlex/eurlex_docs/x_te.npy')

yt_o = np.load('/scratch/work/saxenas2/CVAE_XML/datasets/Eurlex/eurlex_docs/y_te.npy')
y_o = np.load('/scratch/work/saxenas2/CVAE_XML/datasets/Eurlex/eurlex_docs/y_tr.npy')
x_n = np.concatenate((x,x_o),axis=0)
y_n = np.concatenate((y,y_o),axis=0)

import scipy
from scipy.sparse import *
x_n_sp = csr_matrix(x_n)
y_n_sp = csr_matrix(y_n)
from scipy.io import savemat
savemat('../fastxml/manik/x_big_notmanik', {'x_big':x_n_sp})
savemat('../fastxml/manik/y_big_notmanik', {'y_big':y_n_sp})

savemat('../fastxml/manik/x_big_notmanik_test', {'xt':xt_o})
savemat('../fastxml/manik/y_big_notmanik_test', {'yt':yt_o})
savemat('../fastxml/manik/x_from_site', {'x_big':x_o})
savemat('../fastxml/manik/y_from_site', {'y_big':y_o})

