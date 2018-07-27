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


    def kl(self, z_mean, z_log_var):
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + z_mean**2 - 1. - z_log_var, 1))
        return torch.mean(kl_loss)

    def logxy_loss(self, x, x_decoded_mean, params, f=2):
        if(f==0):
            xent_loss = torch.nn.functional.l1_loss(x_decoded_mean, x)*x.shape[-1]
        elif(f==1):
            xent_loss = torch.nn.functional.binary_cross_entropy(x_decoded_mean, x)*x.shape[-1]
        elif(f==2):
            xent_loss = torch.nn.functional.mse_loss(x_decoded_mean, x)*x.shape[-1]
        
        return xent_loss

    def entropy(self, x):
        b = x*torch.log(x+1e-8) + (1-x)*torch.log(1-x+1e-8)
        b = -1.0 * b.mean()*x.shape[-1]
        return b
    
    def cls_loss(self, y, y_pred, params):
        # alpha = 0.1*params.N_unl/params.N
        alpha = 1
        return alpha * torch.nn.functional.binary_cross_entropy(y_pred, y)*y.shape[-1]
