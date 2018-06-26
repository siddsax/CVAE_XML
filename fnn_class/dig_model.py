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
sys.path.append('../utils/')
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
from fnn_model_class import * 
from fnn_test_class import test
from fnn_train_class import train
import sys

def dig(x_tr, y_tr, x_te, y_te, params):
    model = fnn_model_class(params)
    model.load_state_dict(torch.load(params.load_model + "/model_best"))

    all_params = []
    for i,param in enumerate(model.parameters()):
        print(i)
        print(param)
        all_params.append(param.cpu().detach().numpy().tolist())
        print(torch.mean(param))
        print("-------------------")

    fig, ax = plt.subplots()
    # print(all_params)
    ax.hist(all_params, 10)#, weights=num_sold)
    plt.show()