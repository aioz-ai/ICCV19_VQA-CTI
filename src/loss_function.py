'''
Distillation loss
This code is written by Huy Tran
Related paper: https://arxiv.org/pdf/1503.02531.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Distillation_Loss(nn.Module):
    def __init__(self, T, alpha):
        super(Distillation_Loss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.T = T
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, input, knowledge, target):
        loss = nn.KLDivLoss(reduction='none')(nn.functional.log_softmax(input / self.T, dim=1), \
                                              nn.functional.softmax(knowledge / self.T, dim=1)).sum(1).mean() * (
                           self.alpha * self.T * self.T) + (self.bce(input, target) / input.size(0)) * (1. - self.alpha)

        return loss

