import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import sys
import numpy as np
sys.path.append('utils/')
sys.path.append('models/')
import data_helpers 

from w2v import *
from embedding_layer import embedding_layer
from cnn_decoder import cnn_decoder
from cnn_encoder import cnn_encoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
import argparse
from visdom import Visdom
from sklearn.externals import joblib 
from futils import *
from loss import loss

class classifier(nn.Module):
    def __init__(self, params):
        super(classifier, self).__init__()
        self.params = params
        self.l1 = nn.Linear(args.H_dim, args.h_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(args.h_dim, args.y_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(H):
        H = self.l1(H)
        H = self.relu(H)
        H = self.l2(H)
        H = self.sigmoid(H)
        return H