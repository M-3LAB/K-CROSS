import torch
import torch.nn as nn

__all__ = ['NaiveComplexBatchNorm2d', 'ComplexRELU']
class NaiveComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()

        self.bn_real = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_im = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, x):
        output_real = self.bn_real(x.real)
        output_imginary = self.bn_im(x.imag)
        output = torch.complex(output_real, output_imginary)
        return output

class ComplexRELU(nn.Module):
    def __init__(self):
        super(ComplexRELU, self).__init__()

        self.act = nn.ReLU()
    
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
    
