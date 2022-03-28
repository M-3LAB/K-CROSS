import torch
import torch.nn as nn
from model.kaid.complex_nn.fourier_convolve import *
from model.kaid.complex_nn.op import *

__all__ = ['ComplexEncoder', 'ComplexUnetDown']

class ComplexUnetDown(nn.Module):
    def __init__(self, inc, ouc, ks, stride=2, padding=1, slope=0.2):
        super(ComplexUnetDown, self).__init__()
        self.inc = inc
        self.ouc = ouc
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.slope = slope

        self.model = nn.Sequential(
            ComplexConv2d(inc=self.inc, ouc=self.ouc, ks=self.ks, 
                          stride=self.stride, padding=self.padding),
            NaiveComplexBatchNorm2d(num_features=self.ouc),
            ComplexLeakyRELU(slope=self.slope)
        )

    def forward(self, x):
        output = self.model(x)
        return output
class ComplexEncoder(nn.Module):
    def __init__(self):
        super(ComplexEncoder).__init__()

    def forward(self, x):
        pass

