import torch
import torch.nn as nn
from model.kaid.ae.modules.encoder import *
from model.kaid.ae.modules.decoder import *

__all__ = ['ComplexUnet']

class ComplexUnet(nn.Module):

    def __init__(self):
        super(ComplexUnet, self).__init__()
        self.encoder = ComplexEncoder()
        self.decoder = ComplexDecoder()
    
    def encode(self, x):
        z = self.encoder(x) 
        return z

    def forward(self, x):
        z = self.encoder(x) 
        x_hat = self.decode(x)
        return x_hat, z

