import torch
import torch.nn as nn

__all__ = ['FocalFreqLoss']

class FocalFreqLoss(nn.Module):
    def __init__(self):
        super(FocalFreqLoss, self).__init__()
    
    def forward(self, x):
        pass