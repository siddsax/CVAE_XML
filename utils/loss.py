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

class loss:

    def MSLoss(self, X_sample, X):
        t = torch.mean(torch.norm((X_sample - X),1),dim=0) 
        return t
    
    def BCELoss(self, y_pred, y, eps = 1e-100):
        y_pred_1 = torch.log(y_pred+ eps)
        y_pred_2 = torch.log(1 - y_pred + eps)
        t = -torch.sum(torch.mean(y_pred_1*y + y_pred_2*(1-y),dim=0))
        if(t<0):
            print(y_pred)
            print(y_pred_1)
            print(y_pred*y)
            print(y_pred_1*(1-y))
            print(torch.mean(y_pred*y + y_pred_1*(1-y),dim=0))
            print(t)
        return t
    
    def L1Loss(self, X_sample, X):
        t = torch.sum(torch.mean(torch.abs(X_sample - X),dim=0))
        return t
