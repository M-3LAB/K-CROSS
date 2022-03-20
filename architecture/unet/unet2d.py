from architecture.centralized.train import CentralizedTrain
import torch
import torch.nn as nn
from architecture.centralized.train import CentralizedTrain

__all__ = ['Unet2DTrainer']

class Unet2DTrainer(CentralizedTrain):
    def __init__(self, args):
        super(Unet2DTrainer, self).__init__(args=args)
        self.args = args