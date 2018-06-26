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
sys.path.append('../../utils/')
import data_helpers 

from w2v import *
from decoder_classify import decoder
from encoder_classify import encoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
import argparse
from visdom import Visdom
from sklearn.externals import joblib 
from futils import *
from loss import loss

class fnn_model_class(nn.Module):
    def __init__(self, params):
        super(fnn_model_class, self).__init__()
        self.params = params
        self.encoder = encoder(params)
        self.decoder = decoder(params)
        
    def forward(self, batch_x, batch_y):
        # ----------- Encode (X, Y) --------------------------------------------
        # inp = torch.cat([X],1).type(params.dtype)
        z_mu, z_var  = self.encoder.forward(batch_x)
        z = sample_z(z_mu, z_var, self.params)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        # ---------------------------------------------------------------
        
        # ----------- Decode (X, z) --------------------------------------------
        Y_sample = self.decoder.forward(z) 
        recon_loss = self.params.loss_fn(Y_sample, batch_y)
        # ------------------ Check for Recon Loss ----------------------------
        if(recon_loss<0):
            print(recon_loss)
            print(Y_sample[0:100])
            print(batch_y[0:100])
            sys.exit()
        # ---------------------------------------------------------------------

        # ------------ Loss --------------------------------------------------
        loss = self.params.beta*recon_loss + kl_loss
        # --------------------------------------------------------------------

        # return loss.view(-1,1), kl_loss.view(-1,1), recon_loss.view(-1,1)
        return loss, kl_loss, recon_loss
    
    def test(self, X, Y):
        z_mu, z_var  = self.encoder.forward(X)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        # ---------------------------------------------------------------
        
        # ----------- Decode (X, z) --------------------------------------------
        Y_sample = self.decoder.forward(z_mu).data
        recon_loss = self.params.loss_fn(Y_sample, Y)
        loss = self.params.beta*recon_loss + kl_loss
        return Y_sample, loss, kl_loss, recon_loss