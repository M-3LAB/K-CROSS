import torch
import torch.nn as nn

__all__ = ['ConvDropNorm']

class ConvDropNorm(nn.Module):
    def __init__(self, input_channels, output_channels, 
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op =nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropNorm).__init__()
    
    def forward(self, x):
        pass