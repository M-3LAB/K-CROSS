import torch
import torch.nn as nn

__all__ = ['ComplexConv2d', 'ComplexConvTranspose2d', 'ComplexBatchNorm2d']
class ComplexConv2d(nn.Module):
    def __init__(self, inc, ouc, ks, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()

        # Real part convolution 
        self.conv_re = nn.Conv2d(inc, ouc, ks, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        # Imaginary part convolution 
        self.conv_im = nn.Conv2d(inc, ouc, ks, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
    
    def forward(self, x):
        out_real = self.conv_re(x.real) - self.conv_im(x.imag)
        out_imginary = self.conv_im(x.imag) + self.conv_im(x.imag)
        output = torch.complex(out_real, out_imginary)
        return output

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, inc, ouc, ks, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 output_padding=0, padding_modes='zeros'):
        super(ComplexConvTranspose2d, self).__init__()

        # Real part convolution
        self.conv_re = nn.ConvTranspose2d(inc, ouc, ks, stride=stride, padding=padding, 
                                          output_padding=output_padding, groups=groups, bias=bias,
                                          dilation=dilation, padding_modes=padding_modes)

        # Imaginary part convolution
        self.conv_im = nn.ConvTranspose2d(inc, ouc, ks, stride=stride, padding=padding, 
                                          output_padding=output_padding, groups=groups, bias=bias,
                                          dilation=dilation, padding_modes=padding_modes)
    
    def forward(self, x):
        out_real = self.conv_re(x.real) - self.conv_im(x.imag)
        out_imginary = self.conv_im(x.imag) + self.conv_im(x.imag)
        output = torch.complex(out_real, out_imginary)
        return output        

