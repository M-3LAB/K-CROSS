import torch
import torch.nn as nn

__all__ = ['NaiveComplexBatchNorm2d', 'ComplexRELU', 'ComplexLeakyRELU', 'ComplexUpsample']
class NaiveComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine =affine
        self.track_running_stats = track_running_stats

        self.bn_real = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, 
                                      momentum=self.momentum, affine=self.affine, 
                                      track_running_stats=self.track_running_stats)

        self.bn_im = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, 
                                    momentum=self.momentum, affine=self.affine, 
                                    track_running_stats=self.track_running_stats)

    
    def forward(self, x):
        output_real = self.bn_real(x.real)
        output_imginary = self.bn_im(x.imag)
        output = torch.complex(output_real, output_imginary)
        return output

class ComplexRELU(nn.Module):
    def __init__(self, inplace=False):
        super(ComplexRELU, self).__init__()

        self.inplace = inplace
        self.act = nn.ReLU(inplace=self.inplace)
    
    def forward(self, x):
        output_real = self.act(x.real)
        output_imginary = self.act(x.imag)
        output = torch.complex(output_real, output_imginary)
        return output

class ComplexLeakyRELU(nn.Module):
    def __init__(self, slope=0.01):
        super(ComplexLeakyRELU, self).__init__()

        self.act = nn.LeakyReLU(slope)
    
    def forward(self, x):
        output_real = self.act(x.real)
        output_imginary = self.act(x.imag)
        output = torch.complex(output_real, output_imginary)
        return output

class ComplexTanh(nn.Module):
    def __init__(self):
        super(ComplexTanh, self).__init__()
        self.act = nn.Tanh()
    
    def forward(self, x):
        output_real = self.act(x.real)
        output_imginary = self.act(x.imag)
        output = torch.complex(output_real, output_imginary)
        return output

class ComplexUpsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(ComplexUpsample, self).__init__()

        self.scale_factor = scale_factor
        self.act = nn.Upsample(scale_factor=self.scale_factor)

    def forward(self, x):
        output_real = self.act(x.real)
        output_imginary = self.act(x.imag)
        output = torch.complex(output_real, output_imginary)
        return output
    
