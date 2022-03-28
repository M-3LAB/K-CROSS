import torch
import torch.nn as nn

__all__ = ['ComplexBatchNorm2d']
class ComplexBatchNorm2d(nn.Module):
    def __init__(self):
        super(ComplexBatchNorm2d, self).__init__()
    
    def forward(self, x):
        pass

class ComplexRELU(nn.Module):
    def __init__(self):
        super(ComplexRELU).__init__()

        self.act = nn.ReLU()
    
    def forward(self, x):
        output_real = self.act(x.real)
        output_imginary = self.act(x.imag)
        output = torch.complex(output_real, output_imginary)
        return output