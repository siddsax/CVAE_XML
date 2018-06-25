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

class cnn_encoder_decoder(nn.Module):
    def __init__(self, params, embedding_weights):
        super(cnn_encoder_decoder, self).__init__()
        self.params = params
        self.embedding_layer = embedding_layer(params, embedding_weights)
        self.encoder = cnn_encoder(params)
        self.decoder = cnn_decoder(params)
        
    def forward(self, batch_x, batch_y, decoder_word_input, decoder_target):
        # ----------- Encode (X, Y) --------------------------------------------
        e_emb = self.embedding_layer.forward(batch_x)
        z_mu, z_lvar = self.encoder.forward(e_emb, batch_y)
        [batch_size, _] = z_mu.size()
 
        z = Variable(torch.randn([batch_size, self.params.Z_dim])).type(self.params.dtype_f)
        eps = torch.exp(0.5 * z_lvar).type(self.params.dtype_f)
        z = z * eps + z_mu
        
        # -------------------------------------------------------------------------

        # ---------------------------- Decoder ------------------------------------
        decoder_input = self.embedding_layer.forward(decoder_word_input)
        a = 0
        b = 1
        if(self.params.multi_gpu):
            decoder_input = decoder_input.cuda(b)
            batch_y = batch_y.cuda(b)
            z = z.cuda(b)
            logits = self.decoder.forward(decoder_input, z, batch_y)
            logits = logits.view(-1, self.params.vocab_size)
            # kl_loss = (-0.5 * torch.sum(z_lvar - torch.pow(z_mu, 2) - torch.exp(z_lvar) + 1, 1)).mean().squeeze()
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_lvar) + z_mu**2 - 1. - z_lvar, 1))
            cross_entropy = torch.nn.functional.cross_entropy(logits, decoder_target.cuda(2))
            cross_entropy = cross_entropy.cuda(a)
            if(cross_entropy<0):
                print(cross_entropy)
                print(X_sample[0:100])
                print(X[0:100])
                sys.exit()
            loss = self.params.beta*cross_entropy + kl_loss
        else:
            logits = self.decoder.forward(decoder_input, z, batch_y)
            logits = logits.view(-1, self.params.vocab_size)

            kl_loss = (-0.5 * torch.sum(z_lvar - torch.pow(z_mu, 2) - torch.exp(z_lvar) + 1, 1)).mean().squeeze()
            cross_entropy = torch.nn.functional.cross_entropy(logits, decoder_target)
            if(cross_entropy<0):
                print(cross_entropy)
                print(X_sample[0:100])
                print(X[0:100])
                sys.exit()
            loss = self.params.beta*cross_entropy + kl_loss
        
        return loss.view(-1,1), kl_loss.view(-1,1), cross_entropy.view(-1,1)