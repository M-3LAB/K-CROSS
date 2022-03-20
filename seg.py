import torch
import torch.nn as nn
from architecture.unet.unet2d import Unet2DTrainer
from configuration.seg.config import parse_arguments_seg

if __name__ == 'main':
    args = parse_arguments_seg()
