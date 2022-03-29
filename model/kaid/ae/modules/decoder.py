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
        self.inplace = inplace

        self.model = nn.Sequential(
            ComplexConvTranspose2d(inc=self.inc, ouc=self.ouc, ks=self.ks,
                                   stride=self.stride, padding=self.padding),
            NaiveComplexBatchNorm2d(ouc=self.ouc),
            ComplexRELU(inplace=self.inplace)
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        x = self.model(x)
        return x
class ComplexDecoder(nn.Module):
    def __init__(self, ouc_list=[512, 256, 128, 64]):
        super(ComplexDecoder, self).__init__()

        self.ouc_list = ouc_list
        self.up1 = ComplexUnetUp(self.ouc_list[0], self.ouc_list[0]) 
        self.up2 = ComplexUnetUp(self.ouc_list[0]*2, self.ouc_list[1]) 
        self.up3 = ComplexUnetUp(self.ouc_list[1]*2, self.ouc_list[2])
        self.up4 = ComplexUnetUp(self.out_list[2]*2, self.out_list[3])
    
    def forward(self, x):
        pass