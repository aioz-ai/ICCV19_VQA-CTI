"""
QuAMI
Author list
our axiv link

This code is written by Tuong Do.
"""


import torch
import torch.nn as nn

"""Define activation for VQA"""

"""Swish from Searching for Activation Functions
 https://arxiv.org/abs/1710.05941"""
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return torch.mul(x, torch.sigmoid(x))