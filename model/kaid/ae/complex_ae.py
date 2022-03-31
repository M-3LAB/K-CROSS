import torch
import torch.nn as nn
from model.kaid.ae.modules.encoder import *
from model.kaid.ae.modules.decoder import *

__all__ = ['ComplexUnet']

class ComplexUnet(nn.Module):

    def __init__(self, encode_ouc_list=[64, 128, 256, 512, 512],
                 decode_ouc_list=[512, 256, 128, 64]):
        super(ComplexUnet, self).__init__()

        self.encode_ouc_list = encode_ouc_list
        self.decode_ouc_list = decode_ouc_list

        self.down1 = ComplexUnetDown(inc=1, ouc=self.encode_ouc_list[0])
        self.down2 = ComplexUnetDown(inc=self.encode_ouc_list[0], ouc=self.encode_ouc_list[1])
        self.down3 = ComplexUnetDown(inc=self.encode_ouc_list[1], ouc=self.encode_ouc_list[2])
        self.down4 = ComplexUnetDown(inc=self.encode_ouc_list[2], ouc=self.encode_ouc_list[3])
        self.down5 = ComplexUnetDown(inc=self.encode_ouc_list[3], ouc=self.encode_ouc_list[4])

        self.up1 = ComplexUnetUp(inc=self.decode_ouc_list[0], ouc=self.decode_ouc_list[0]) 
        self.up2 = ComplexUnetUp(inc=self.decode_ouc_list[0]*2, ouc=self.decode_ouc_list[1]) 
        self.up3 = ComplexUnetUp(inc=self.decode_ouc_list[1]*2, ouc=self.decode_ouc_list[2])
        self.up4 = ComplexUnetUp(inc=self.decode_ouc_list[2]*2, ouc=self.decode_ouc_list[3])

        self.final = ComplexFinalLayer(inc=self.decode_ouc_list[3]*2, ouc=1)
    
    def encode(self, x):
        self.d1 = self.down1(x)
        self.d2 = self.down2(self.d1)
        self.d3 = self.down3(self.d2)
        self.d4 = self.down4(self.d3)
        z = self.down5(self.d4)
        return z 
    
    def decode(self, z):
        self.u1 = self.up1(z)
        self.u2 = self.up2(self.u1, self.d4)
        self.u3 = self.up3(self.u2, self.d3)
        self.u4 = self.up4(self.u3, self.d2)
        x_hat = self.final(self.u4, self.d1)
        return x_hat

    def forward(self, x):
        z = self.encode(x) 
        x_hat = self.decode(z)
        return x_hat, z

