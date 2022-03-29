from model.kaid.complex_nn.op import ComplexTanh
import torch
import torch.nn as nn
from model.kaid.complex_nn.fourier_convolve import *
from model.kaid.complex_nn.op import *

__all__ = ['ComplexDecoder', 'ComplexUnetUp', 'ComplexFinalLayer']

class ComplexFinalLayer(nn.Module):
    def __init__(self, inc, ouc, ks=3, scale_factor=2, padding=1):
        super(ComplexFinalLayer, self).__init__()

        self.inc = inc
        self.ouc = ouc
        self.scale_factor = scale_factor
        self.ks = ks
        self.padding = padding

        self.model = nn.Sequential(
            ComplexUpsample(scale_factor=self.scale_factor),
            ComplexConv2d(inc=self.inc, ouc=self.ouc, ks=self.ks, padding=self.padding),
            ComplexTanh()
        )
    
    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        x = self.model(x)
        return x 

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

        self.final = ComplexFinalLayer(inc=self.ouc_list[3]*2, ouc=1)
    
    def forward(self, z):
        self.u1 = self.up1(z)
        self.u2 = self.up2(self.u1, self.d4)
        self.u3 = self.up3(self.u2, self.d3)
        self.u4 = self.up4(self.u3, self.d2)
        x_hat = self.final(self.u4, self.d1)
        return x_hat