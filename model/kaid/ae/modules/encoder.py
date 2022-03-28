import torch
import torch.nn as nn
from model.kaid.complex_nn.fourier_convolve import *
from model.kaid.complex_nn.op import *

__all__ = ['ComplexEncoder', 'ComplexUnetDown']

class ComplexUnetDown(nn.Module):
    def __init__(self, inc, ouc, ks=3, stride=2, padding=1, slope=0.2):
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
    def __init__(self, ouc_list=[64, 128, 256, 512, 512]):
        super(ComplexEncoder).__init__()
        self.ouc_list = ouc_list
        self.down1 = ComplexUnetDown(1, 64)
        self.down2 = ComplexUnetDown(64, 128)
        self.down3 = ComplexUnetDown(128, 256)
        self.down4 = ComplexUnetDown(256, 512)
        self.down5 = ComplexUnetDown(512, 512)

    def forward(self, x):
        self.d1 = self.down1(x)
        self.d2 = self.down2(self.d1)
        self.d3 = self.down3(self.d2)
        self.d4 = self.down4(self.d3)
        z = self.down5(self.d4)
        return z
        
        

