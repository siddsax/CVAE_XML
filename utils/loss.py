import os
import sys
import torch
import timeit
import argparse
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import pdb

def isnan(x):
    return x != x

class loss:

    def MSLoss(self, X_sample, X):
        x = (X_sample - X)
        t = torch.mean(torch.sum(x*x,dim=1))
        # t = torch.mean(torch.norm((X_sample - X),1),dim=0) 
        return t
    
    def BCELoss(self, y_pred, y, eps = 1e-25):
        # y_pred_1 = torch.log(y_pred+ eps)
        # y_pred_2 = torch.log(1 - y_pred + eps)
        # t = -torch.sum(torch.mean(y_pred_1*y + y_pred_2*(1-y),dim=0))
        t = torch.nn.functional.binary_cross_entropy(y_pred, y)*y.shape[-1]
        if(torch.__version__=='.0.4.0'):
		if(torch.isnan(t).any()):
            		print("nan")
           		pdb.set_trace()
        	if(t<0):
            		print("negative")            
           		pdb.set_trace()
        return t
    
    def L1Loss(self, X_sample, X):
        t = torch.mean(torch.sum(torch.abs(X_sample - X),dim=1))
        return t

    def kl(self, z_mean, z_log_var):
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + z_mean**2 - 1. - z_log_var, 1))
        
        return torch.mean(kl_loss)

    def logxy_loss(self, x, x_decoded_mean, params):
        # xent_loss = torch.nn.functional.binary_cross_entropy(x_decoded_mean, x)*x.shape[-1]
        xent_loss = torch.nn.functional.mse_loss(x_decoded_mean, x)*x.shape[-1]
    
        # p(y) for observed data is equally distributed
        logy = torch.from_numpy(np.array([np.log(1. / params.y_dim)])).type(params.dtype)
        
        return xent_loss - logy

    def cls_loss(self, y, y_pred, params):
        alpha = 0.1 * params.N
        return alpha * torch.nn.functional.binary_cross_entropy(y_pred, y)

    # def unlabeled_vae_loss(self,x, x_decoded_mean):
    #     entropy = metrics.categorical_crossentropy(_y_output, _y_output)
    #     labeled_loss = logxy_loss(x, x_decoded_mean) + kl(x, x_decoded_mean)
        
    #     return K.mean(K.sum(_y_output * labeled_loss, axis=-1)) + entropy