from turtle import forward
import torch
import torch.nn as nn
from model.kaid.ae.modules.encoder import *
from model.kaid.ae.modules.decoder import *

__all__ = ['ComplexUnet']

class ComplexUnet(nn.Module):
    def __init__(self):
        super(ComplexUnet).__init__()
        self.encoder = ComplexEncoder()
    
    def forward(self, x):
        pass

