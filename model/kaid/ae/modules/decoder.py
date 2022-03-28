import torch
import torch.nn as nn
from model.kaid.complex_nn.fourier_convolve import *
from model.kaid.complex_nn.op import *

__all__ = ['ComplexDecoder', 'ComplexUnetUp']
class ComplexUnetUp(nn.Module):
    def __init__(self, inc, ouc, ks, stride=2, padding=1, inplace=True):
        super(ComplexUnetUp, self).__init__()
        self.inc = inc
        self.ouc = ouc
        self.ks = ks
        self.stride = stride
        self.padding = padding

        self.model = nn.Sequential(
            ComplexConvTranspose2d(),
            NaiveComplexBatchNorm2d(),
            ComplexRELU()
        )

    def forward(self, x):
        pass
class ComplexDecoder(nn.Module):
    def __init__(self):
        super(ComplexDecoder).__init__()
    
    def forward(self, x):
        pass