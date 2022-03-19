import torch
import torch.nn as nn
from segmentation.base import SegmentationNetwork
from segmentation.modules.block import ConvDropNorm
from segmentation.modules.stackedconv import StackedConvLayers
from segmentation.common import softmax_helper, InitWeights_He

class UNet2D(SegmentationNetwork):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                conv_kernel_sizes=None,
                upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                max_num_features=None, basic_block=ConvDropNorm,
                seg_output_use_bias=False):

        super(UNet2D, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
    
