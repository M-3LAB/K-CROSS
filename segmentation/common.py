import torch
import torch.nn.functional as F

__all__ = ['no_op', 'maybe_to_torch', 'to_cuda', 'softmax_helper']

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

