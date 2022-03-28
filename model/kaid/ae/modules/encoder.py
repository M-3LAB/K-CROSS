import torch
import torch.nn as nn
from model.kaid.complex_nn.fourier_convolve import *
from model.kaid.complex_nn.op import *

__all__ = ['ComplexEncoder', 'ComplexUnetDown']

class ComplexUnetDown(nn.Module):
    def __init__(self, inc, ouc, ks, stride=2, padding=1):
        super(ComplexUnetDown, self).__init__()
        self.model = nn.Sequential(
            ComplexConv2d(inc=inc, ouc=ouc, ks=ks, stride=stride, padding=padding),
            NaiveComplexBatchNorm2d(num_features=ouc),
            ComplexLeakyRELU(slope=0.2)
        )

    def forward(self, x):
        output = self.model(x)
        return output
class ComplexEncoder(nn.Module):
    def __init__(self):
        super(ComplexEncoder).__init__()

    def forward(self, x):
        pass

