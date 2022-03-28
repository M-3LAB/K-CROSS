import torch
import torch.nn as nn

__all__ = ['ComplexDecoder', 'ComplexUnetUp']
class ComplexUnetUp(nn.Module):
    def __init__(self, inc, ouc, ks, stride=2, padding=1, inplace=True):
        super(ComplexUnetUp, self).__init__()
        self.inc = inc
        self.ouc = ouc
        self.ks = ks
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        pass
class ComplexDecoder(nn.Module):
    def __init__(self):
        super(ComplexDecoder).__init__()
    
    def forward(self, x):
        pass