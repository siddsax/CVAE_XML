from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel_o(params, shape, eps=1e-20):
    U = torch.rand(shape).type(params.dtype)
    return - Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample_o(params, logits, temperature):
    logits = torch.cat((torch.log(logits), torch.log(1-logits)), -1)
    y = logits + sample_gumbel_o(params, logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax_o(params, logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(params, logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y, y

def gumbel_multiSample_o(params, logits, temperature):
    sample = []
    #print(logits.shape)
    for i in range(logits.shape[-1]):
        A = gumbel_softmax_sample_o(params, (logits[:, i]).contiguous().view((-1, 1)), temperature)
	sample.append(A[:, 0].contiguous().view((-1, 1)))
    sample = torch.cat(sample, dim=-1)
    return sample

def sample_gumbel(params, shape, eps=1e-20):
    U = torch.rand(shape).type(params.dtype)
    return - Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(params, logits, temperature):
    # logits = torch.cat((logits, 1-logits), -1)
    y = logits + sample_gumbel(params, logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=-1)[:, :, 0]

# def gumbel_softmax(params, logits, temperature):
#     """
#     input: [*, n_class]
#     return: [*, n_class] an one-hot vector
#     """
#     y = gumbel_softmax_sample(params, logits, temperature)
#     shape = y.size()
#     _, ind = y.max(dim=-1)
#     y_hard = torch.zeros_like(y).view(-1, shape[-1])
#     y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     return (y_hard - y).detach() + y, y

def gumbel_multiSample(params, logits, temperature):
    sample = []
    #print(logits.shape)
    logits = logits.view(logits.shape[0], logits.shape[1], 1)
    logits = torch.cat((torch.log(logits), torch.log(1-logits)), dim=-1)
    sample = gumbel_softmax_sample(params, logits, temperature)
    # for i in range(logits.shape[-1]):
    #     A = gumbel_softmax_sample(params, (logits[:, i]).contiguous().view((-1, 1)), temperature)
    #     sample.append(A[:, 0].contiguous().view((-1, 1)))
    # sample = torch.cat(sample, dim=-1)
    return sample
@profile
def main():
    import math
    import sys
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--t', dest='temp', type=float, default=.8, help='Temperature')
    params = parser.parse_args()
    params.dtype = torch.FloatTensor
    #print(gumbel_softmax(Variable(torch.FloatTensor([[0.1), 0.4), 0.3), 0.2)]] * 20000)),     0.8).sum(dim=0))
    #print(gumbel_softmax(Variable(torch.FloatTensor([[0.1), 0.4), 0.3), 0.2)]] * 2)),     0.8))
    a = list(np.random.rand(5000))
    sample = gumbel_multiSample(params, Variable(torch.FloatTensor([a] * 1)), params.temp)
    #print(sample)
    
    sample = gumbel_multiSample_o(params, Variable(torch.FloatTensor([a] * 1)), params.temp)
    #print(sample)

main()
