import os
import torch
import random
import cPickle
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
from w2v import load_word2vec
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.gridspec as gridspec

def out_size(l_in, padding, dilation, kernel_size, stride):
    a = l_in + 2*padding - dilation*(kernel_size - 1) -1
    b = a/stride
    return b + 1

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class CNN_encoder(torch.nn.Module):

    def __init__(self, args):
        # Model Hyperparameters
        super(CNN_encoder, self).__init__()
        
        # Model Hyperparameters
        self.sequence_length = args.sequence_length
        self.embedding_dim = args.embedding_dim
        self.filter_sizes = args.filter_sizes
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.pooling_type = args.pooling_type
        self.hidden_dims = args.hidden_dims

        # Training Hyperparameters
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs

        # Model variation and w2v pretrain type
        self.model_variation = args.model_variation        # CNN-rand | CNN-pretrain
        self.pretrain_type = args.pretrain_type

        # Fix random seed
        # np.random.seed(1126)
        # random.seed(1126)
        # self.rng = T.shared_randomstreams.RandomStreams(seed=1126)

        self.l0 = nn.Linear(self.vocab_size, self.embedding_dim, bias=True)
        if self.model_variation == 'pretrain':
            self.l0.weights = self.embedding_weights
        self.drp0 = mm.Dropout(p=.25)

        self.conv_layers = []
        self.pool_layers = []
        for fsz in self.filter_sizes:
            l_conv = nn.Conv1d(1, self.num_filters, fsz, stride=2, bias=True)
            self.conv_layers.append(l_conv)
            l_out_size = out_size(l_in, 0, 1, fsz, 2)
            pool_size = l_out_size // self.pooling_units
            
            if self.pooling_type == 'average':
                l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
                # l_pool = lasagne.layers.Pool1DLayer(l_conv, pool_size, stride=None, mode='average_inc_pad')
            elif self.pooling_type == 'max':
                l_pool = nn.MaxPool1d(2, stride=1)

            self.pool_layers.append(l_pool)
    
    def forward(self, inputs):

        o = self.l0(inputs)
        o0 = self.drp0(o)

        conv_out = []
        for fsz in range(len(self.filter_sizes)):
            o = self.conv_layers[i](o0)
            o = self.pool_layers[i](o)
            o = o.view(-1,1)
            conv_out.append(o)

        # for fsz in self.filter_sizes:
        #     l_conv = lasagne.layers.Conv1DLayer(l_embed_dropout,
        #                                         num_filters=self.num_filters,
        #                                         filter_size=fsz)
        #     l_conv_shape = lasagne.layers.get_output_shape(l_conv)
        #     pool_size = l_conv_shape[-1] // self.pooling_units
        #     if self.pooling_type == 'average':
        #         l_pool = lasagne.layers.Pool1DLayer(l_conv, pool_size, stride=None, mode='average_inc_pad')
        #     elif self.pooling_type == 'max':
        #         l_pool = lasagne.layers.MaxPool1DLayer(l_conv, 2, stride=1)
        #     else:
        #         raise NotImplementedError('Unknown pooling_type!')
            # l_flat = lasagne.layers.flatten(l_pool)
            # convs.append(l_flat)
        if len(self.filter_sizes)>1:
            conv_out = torch.cat(conv_out)
        else:
            conv_out = convs[0]

        return conv_out
        # Final hidden layer
        # l_hidden = lasagne.layers.DenseLayer(l_conv_final, num_units=self.hidden_dims, nonlinearity=lasagne.nonlinearities.rectify)
        # l_hidden_dropout = lasagne.layers.DropoutLayer(l_hidden, p=0.5)

        # l_y = lasagne.layers.DenseLayer(l_hidden_dropout, num_units=self.output_dim, nonlinearity=lasagne.nonlinearities.sigmoid)
        # params = lasagne.layers.get_all_params(l_y, trainable=True)
        # self.network = l_y

        # # Objective function and update params
        # Y_pred = lasagne.layers.get_output(l_y)
        # loss = lasagne.objectives.binary_crossentropy(Y_pred, Y).mean()
        # updates = lasagne.updates.adam(loss, params)
        # self.train_fn = theano.function([inputs, Y], [loss], updates=updates)

