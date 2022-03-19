import torch
import torch.nn as nn
from segmentation.base import SegmentationNetwork
from segmentation.modules.block import ConvDropNorm
from segmentation.modules.stackedconv import StackedConvLayers

class UNet2D(SegmentationNetwork):
    def __init__(self):
        pass
