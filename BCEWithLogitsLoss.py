import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import sys

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(_WeightedLoss, self).__init__(size_average, reduce)
        self.register_buffer('weight', weight)



class BCEWithLogitsLoss(_Loss):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ t_n \cdot \log \sigma(x_n)
        + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`N` is the batch size. If reduce is ``True``, then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over each loss element in the batch. Note that for some losses, there
            multiple elements per sample. If the field :attr:`size_average` is set to
            ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True

     Shape:
         - Input: :math:`(N, *)` where `*` means, any number of additional
           dimensions
         - Target: :math:`(N, *)`, same shape as the input

     Examples::

        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(BCEWithLogitsLoss, self).__init__(size_average, reduce)
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            return F.binary_cross_entropy_with_logits(input, target,
                                                      self.weight,
                                                      self.size_average,
                                                      reduce=self.reduce)
        else:
            return F.binary_cross_entropy_with_logits(input, target,
                                                      size_average=self.size_average,
                                                      reduce=self.reduce)