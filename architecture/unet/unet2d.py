from architecture.centralized.train import CentralizedTrain
import torch
import torch.nn as nn
from architecture.centralized.train import CentralizedTrain
import yaml

__all__ = ['Unet2DTrainer']

class Unet2DTrainer(CentralizedTrain):
    def __init__(self, args):
        super(Unet2DTrainer, self).__init__(args=args)
        self.args = args
        assert self.args.dataset == 'brats2021', 'Only BraTS for Segmentation'
    
    def load_config(self):
       # load dataset 
       with open('./configuration/seg/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_seg = yaml.load(f, Loader=yaml.SafeLoader)