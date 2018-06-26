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
sys.path.append('utils/')
sys.path.append('models/')
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
from encoder_classify import encoder
from decoder_classify import decoder
import argparse
from visdom import Visdom
from sklearn.externals import joblib 
from futils import *
from loss import loss
from fnn_model_class import fnn_model_class 

def test(x_te, y_te, params, model=None, best_test_loss=None):

    if(model==None):
        model = fnn_model_class(params)
        model.load_state_dict(torch.load("saved_models/" + params.model_name + "/model_best"))#, map_location=lambda storage, loc: storage)
    
    model.eval()

    X, Y = load_data(x_te, y_te, params, batch=False)
    Y_sample, loss, kl_loss, recon_loss = model.test(X, Y)

    if(best_test_loss!=None):
        if(loss < best_test_loss ):
            best_test_loss = loss
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(loss, kl_loss, recon_loss, best_test_loss))
        return best_test_loss
    else:
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4};'.format(loss.data, kl_loss.data, recon_loss.data))
        Y_probabs = sparse.csr_matrix(Y_sample.data)
        sio.savemat('score_matrix.mat' , {'score_matrix': Y_probabs})
