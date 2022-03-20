from architecture.centralized.train import CentralizedTrain
import torch
import torch.nn as nn
from architecture.centralized.train import CentralizedTrain
import yaml
from tools.utilize import *

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
        # load basic 
        with open('./configuration/architecture/3_dataset_base/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_dataset = yaml.load(f, Loader=yaml.SafeLoader)
        with open('configuration/architecture/2_train_base/centralized_training.yaml', 'r') as f:
            config_train = yaml.load(f, Loader=yaml.SafeLoader)
        with open('configuration/architecture/1_model_base/{}.yaml'.format(self.args.model), 'r') as f:
            config_model = yaml.load(f, Loader=yaml.SafeLoader)
        
        config = override_config(config_model, config_train)
        config = override_config(config, config_dataset)
        config = override_config(config, config_seg)
        self.para_dict = merge_config(config, self.args)
        self.args = extract_config(self.args)