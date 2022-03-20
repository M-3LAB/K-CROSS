import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

__all__ = ['no_op', 'maybe_to_torch', 'to_cuda', 'softmax_helper', 'InitWeights_He',
           'sum_tensor']

class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def maybe_to_torch(data):
    if isinstance(data, list):
        data = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in data]
    elif not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    return data


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data

softmax_helper = lambda x: F.softmax(x, 1)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

