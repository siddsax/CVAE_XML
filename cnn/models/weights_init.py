import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

def weights_init(m):
    if(torch.__version__=='0.4.0'):
        torch.nn.init.xavier_uniform_(m)
    else:
        torch.nn.init.xavier_uniform(m)
