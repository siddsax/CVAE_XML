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
        logy = Variable(torch.from_numpy(np.array([np.log(1. / params.y_dim)])).type(params.dtype))
        
        return xent_loss - logy

    def entropy(self, x):
        b = x*torch.log(x+1e-8) + (1-x)*torch.log(1-x+1e-8)

        # b = torch.nn.functional.softmax(x, dim=-1) * torch.nn.functional.log_softmax(x, dim=-1)
        b = -1.0 * b.mean()*x.shape[-1]
        # pdb.set_trace()
        return b
    
    def cls_loss(self, y, y_pred, params):
        alpha = 1.0
        # alpha = 0.1 * params.N
        return alpha * torch.nn.functional.mse_loss(y_pred, y)*y.shape[-1]
        # return alpha * torch.nn.functional.binary_cross_entropy(y_pred, y)*y.shape[-1]
        # return alpha * torch.mean(torch.sum(torch.abs(y_pred - y),dim=1))
        # return alpha * self.ranking_mse_loss(y, y_pred, params)
    def ranking_mse_loss(self, y, y_pred, params):
        alpha = 0.1 * params.N

        # x = torch.randn(10)
        # y, ind = torch.sort(x, 1)
        # unsorted = y.new(*y.size())
        # unsorted.scatter_(1, ind, y)
        # print((x - unsorted).abs().max())

        rank_mat = np.argsort(y_pred.data.cpu().numpy())
        # rank_mat = np.argsort(y_pred)
        # backup = np.copy(y_pred)
        # for k in range(k):
        v1 = y_pred.clone()
        v2 = y.clone()
        for i in range(y_pred.shape[0]):
            for k in range(5):
                v1[i,rank_mat[i, -(k+1)]] = ((k+1)**2)*y_pred[i,rank_mat[i, -(k+1)]].clone()
                # y_pred[i,rank_mat[i, :-(k+1)]] = m
                # m1 = 
                v2[i, rank_mat[i, -(k+1)]] = ((k+1)**2)*y[i,rank_mat[i, -(k+1)]].clone()
        return torch.nn.functional.mse_loss(v1, v2)*y.shape[-1]
        # y_pred = np.ceil(y_pred)
        # kk = np.argwhere(y_pred>0)
        # mat = np.multiply(y_pred, y)
        # num = np.sum(mat,axis=1)
        # p[k] = np.mean(num/(k+1))
    # def unlabeled_vae_loss(self,x, x_decoded_mean):
    #     entropy = metrics.categorical_crossentropy(_y_output, _y_output)
    #     labeled_loss = logxy_loss(x, x_decoded_mean) + kl(x, x_decoded_mean)
        
    #     return K.mean(K.sum(_y_output * labeled_loss, axis=-1)) + entropy


