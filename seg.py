import torch
import yaml
import os
import numpy as np

from torch.utils.data import DataLoader
from data_io.ixi import IXI
from data_io.brats import BraTS2021
from architecture.unet.unet2d import Unet2DTrainer
from configuration.seg.config import parse_arguments_seg
from tools.utilize import *

if __name__ == '__main__':
    args = parse_arguments_seg()
    with open('./configuration/kaid/{}.yaml'.format(args.dataset), 'r') as f:
        para_dict = yaml.load(f, Loader=yaml.SafeLoader)
    para_dict = merge_config(para_dict, args)
    print(para_dict)

    file_path = record_path(para_dict)
    if para_dict['save_log']:
        save_arg(para_dict, file_path)
        save_script(__file__, file_path)

    with open('./work_dir/log_running.txt'.format(file_path), 'a') as f:
        print('---> {}'.format(file_path), file=f)
        print(para_dict, file=f)

    device, device_ids = parse_device_list(para_dict['gpu_ids'], 
                                           int(para_dict['gpu_id'])) 

    seed_everything(para_dict['seed'])

    normal_transform = [{'degrees':0, 'translate':[0.00, 0.00],
                         'scale':[1.00, 1.00], 
                         'size':(para_dict['size'], para_dict['size'])},
                        {'degrees':0, 'translate':[0.00, 0.00],
                         'scale':[1.00, 1.00], 
                         'size':(para_dict['size'], para_dict['size'])}]

    train_dataset = BraTS2021(root=para_dict['data_path'],
                              modalities=[para_dict['source_domain'], para_dict['target_domain']],
                              extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                              noise_type='normal',
                              learn_mode='train', # train or test is meaningless if dataset_spilited is false
                              transform_data=normal_transform,
                              data_mode='paired',
                              data_num=para_dict['pair_num'],
                              dataset_splited=True,
                              annotation=para_dict['segmentaion'])

    test_dataset = BraTS2021(root=para_dict['data_path'],
                              modalities=[para_dict['source_domain'], para_dict['target_domain']],
                              noise_type=para_dict['noise_type'],
                              learn_mode='test',
                              extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                              transform_data=normal_transform,
                              data_mode='paired',
                              data_num=para_dict['pair_num'],
                              dataset_splited=True,
                              annotation=para_dict['segmentaion'])
        
    train_loader = DataLoader(train_dataset, num_workers=para_dict['num_workers'],
                                batch_size=para_dict['batch_size'], shuffle=False)

    test_loader = DataLoader(test_dataset, num_workers=para_dict['num_workers'],
                                batch_size=para_dict['batch_size'], shuffle=False)
