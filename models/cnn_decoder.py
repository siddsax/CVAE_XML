import os
import sys
import torch
import random

import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
sys.path.insert(0, '../utils')

from w2v import load_word2vec
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.gridspec as gridspec
from torch.nn import Parameter

class cnn_decoder(nn.Module):
    def __init__(self, params):
        super(cnn_decoder, self).__init__()

        self.params = params
        # Params Init
        self.kernels = [Parameter(torch.Tensor(out_chan, in_chan, width).normal_(0, 0.05))
                        for out_chan, in_chan, width in params.decoder_kernels]
        self.biases = [Parameter(torch.Tensor(out_chan).normal_(0, 0.05))
                       for out_chan, in_chan, width in params.decoder_kernels]

        self._add_to_parameters(self.kernels, 'decoder_kernel')
        self._add_to_parameters(self.biases, 'decoder_bias')


        self.out_size = self.params.decoder_kernels[-1][0]
        self.fc = nn.Linear(self.out_size, self.params.vocab_size)

    def forward(self, decoder_input, z, drop_prob):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence latent variable with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :return: unnormalized logits of sentense words distribution probabilities
                 with shape of [batch_size, seq_len, word_vocab_size]
        """

        [batch_size, seq_len, embed_size] = decoder_input.size()

        '''
            decoder is conditioned on context via additional bias = W_cond * z to every input token
        '''

        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.Z_dim)
        decoder_input = torch.cat([decoder_input, z], 2)
        decoder_input = nn.Dropout(decoder_input, drop_prob)

        # x is tensor with shape [batch_size, input_size=in_channels, seq_len=input_width]
        x = decoder_input.transpose(1, 2).contiguous()

        for layer, kernel in enumerate(self.kernels):
            # apply conv layer with non-linearity and drop last elements of sequence to perfrom input shifting
            x = nn.conv1d(x, kernel,
                         bias=self.biases[layer],
                         dilation=self.params.decoder_dilations[layer],
                         padding=self.params.decoder_paddings[layer])

            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.params.decoder_paddings[layer])].contiguous()

            x = nn.relu(x)


        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, self.out_size)
        x = self.fc(x)
        result = x.view(-1, seq_len, self.params.vocab_size)

        return result

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)

