from locale import normalize
from tools.visualize import np_scaling_kspace, torch_scaling_kspace
import torch
import yaml
import os
import numpy as np

from torch.utils.data import DataLoader
from data_io.ixi import IXI
from data_io.brats import BraTS2021
from tools.utilize import *
from model.cyclegan.cyclegan import CycleGen 
from model.munit.munit import Encoder as MUE
from model.munit.munit import Decoder as MUD
from model.unit.unit import Encoder as UE 
from model.unit.unit import Generator as UG

from configuration.kaid.config import parse_arguments_kaid
from loss_function.kaid.distance import l1_diff, l2_diff, cosine_similiarity
from model.kaid.FT.fourier_transform import * 
from model.kaid.FT.power_spectrum import *
from metrics.kaid.stats import mask_stats, best_radius_list 
from model.kaid.ae.kaid_ae import KAIDAE

from tools.visualize import *

if __name__ == '__main__':
    args = parse_arguments_kaid()
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

    kaid_transform = [{'size':(para_dict['size'], para_dict['size'])},
                      {'size':(para_dict['size'], para_dict['size'])}]

    if para_dict['noise_type'] == 'gaussian':
        noise_transform = [{'mu':para_dict['a_mu'], 'sigma':para_dict['a_sigma'],
                            'size':(para_dict['size'], para_dict['size'])},
                           {'mu':para_dict['b_mu'], 'sigma':para_dict['b_sigma'],
                            'size':(para_dict['size'], para_dict['size'])}]

    elif para_dict['noise_type'] == 'slight':
        noise_transform = [{'degrees': para_dict["a_rotation_degrees"],
                            'translate': [para_dict['a_trans_lower_limit'], para_dict['a_trans_upper_limit']],
                            'scale': [para_dict['a_scale_lower_limit'], para_dict['a_scale_upper_limit']],
                            'size': (para_dict['size'], para_dict['size']),'fillcolor': 0},
                           {'degrees': para_dict['b_rotation_degrees'],
                            'translate': [para_dict['b_trans_lower_limit'], para_dict['b_trans_upper_limit']],
                            'scale': [para_dict['b_scale_lower_limit'], para_dict['b_scale_uppper_limit']],
                            'size': (para_dict['size'], para_dict['size']),'fillcolor': 0}]

    elif para_dict['noise_type'] == 'severe':
        noise_transform = [{'degrees':para_dict['severe_rotation'], 
                            'translate':[para_dict['severe_translation'], para_dict['severe_translation']],
                            'scale':[1-para_dict['severe_scaling'], 1+para_dict['severe_scaling']], 
                            'size':(para_dict['size'], para_dict['size'])},
                            {'degrees':para_dict['severe_rotation'], 
                             'translate':[para_dict['severe_translation'], para_dict['severe_translation']],
                             'scale':[1-para_dict['severe_scaling'], 1+para_dict['severe_scaling']], 
                             'size':(para_dict['size'], para_dict['size'])}]
    else:
        raise NotImplementedError('New Noise Has Not Been Implemented')
    
    #Dataset IO
    if para_dict['dataset'] == 'ixi':
        assert para_dict['source_domain'] in ['t2', 'pd']
        assert para_dict['target_domain'] in ['t2', 'pd']
    
        ixi_normal_dataset = IXI(root=para_dict['data_path'],
                                 modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                 extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                 noise_type='normal',
                                 learn_mode='train', #train or test is meaningless if dataset_splited is false
                                 transform_data=normal_transform,
                                 data_mode='paired',
                                 data_num=para_dict['data_num'],
                                 dataset_splited=False)
        
        ixi_noise_dataset = IXI(root=para_dict['data_path'],
                                modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                noise_type=para_dict['noise_type'],
                                learn_mode='train', #train or test is meaningless if dataset_splited is false
                                transform_data=noise_transform,
                                data_mode='paired',
                                data_num=para_dict['data_num'],
                                dataset_splited=False)
        
        ixi_test_dataset = IXI(root=para_dict['data_path'],
                                 modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                 extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                 noise_type='kaid',
                                 learn_mode='train', #train or test is meaningless if dataset_splited is false
                                 transform_data=kaid_transform,
                                 data_mode='paired',
                                 data_num=para_dict['data_num'],
                                 dataset_splited=False,
                                 assigned_images=para_dict['assigned_images'])

        #TODO: make sure normal and nosiy loader release the same order of dataset
        normal_loader = DataLoader(ixi_normal_dataset, num_workers=para_dict['num_workers'],
                                   batch_size=para_dict['batch_size'], shuffle=False)

        noisy_loader = DataLoader(ixi_noise_dataset, num_workers=para_dict['num_workers'],
                                  batch_size=para_dict['batch_size'], shuffle=False)

        test_loader = DataLoader(ixi_test_dataset, num_workers=para_dict['num_workers'],
                                 batch_size=2, shuffle=False)

        
    elif para_dict['dataset'] == 'brats2021':
        assert para_dict['source_domain'] in ['t1', 't2', 'flair']
        assert para_dict['target_domain'] in ['t1', 't2', 'flair']

        """
        #TODO: Create a dataset contained the whole part of BraTS 2021, included training and validation 
        """
        
        brats_normal_dataset = BraTS2021(root=para_dict['data_path'],
                                         modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                         extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                         noise_type='normal',
                                         learn_mode='train', # train or test is meaningless if dataset_spilited is false
                                         transform_data=normal_transform,
                                         data_mode='paired',
                                         data_num=para_dict['data_num'])

        brats_noise_dataset = BraTS2021(root=para_dict['data_path'],
                                        modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                        noise_type=para_dict['noise_type'],
                                        learn_mode='train',
                                        extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                        transform_data=noise_transform,
                                        data_mode='paired',
                                        data_num=para_dict['data_num'])
        
        #TODO: make sure normal and nosiy loader release the same order of dataset
        normal_loader = DataLoader(brats_normal_dataset, num_workers=para_dict['num_workers'],
                                   batch_size=para_dict['batch_size'], shuffle=False)

        noisy_loader = DataLoader(brats_noise_dataset, num_workers=para_dict['num_workers'],
                                  batch_size=para_dict['batch_size'], shuffle=False)
        
        test_loader = DataLoader(brats_normal_dataset, num_workers=para_dict['num_workers'],
                                 batch_size=1, shuffle=False)
    else:
        raise NotImplementedError("New Data Has Not Been Implemented")

    # Debug Mode
    if para_dict['debug']:
        batch_limit = 1
    else:
        batch_limit = int(para_dict['data_num'] / para_dict['batch_size'])

    # Model
    kaid_ae = KAIDAE().to(device)
    # Loss
    criterion_recon = torch.nn.L1Loss().to(device)
    criterion_high_freq = torch.nn.MSELoss(reduction='mean').to(device)
    criterion_low_freq = torch.nn.MSELoss(reduction='mean').to(device)

    # Optimizer
    optimizer = torch.optim.Adam(kaid_ae.parameters(), lr=para_dict['lr'],
                                 betas=[para_dict['beta1'], para_dict['beta2']])

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=para_dict['step_size'],
                                                   gamma=para_dict['gamma']) 
    
    # Mask Statistics  
    """
    src: source domain
    tag: target domain
    msl: the half of mask side length
    """ 
    radius_path = os.path.join(para_dict['radius_path'], para_dict['dataset']) 
    create_folders(radius_path) 

    if para_dict['radius_stats']:
        assert para_dict['radius_assigned'] is False, 'msl_stats and msl_assigned are contradictory'
        src_dict, tag_dict = mask_stats(normal_loader, para_dict['source_domain'], 
                                        para_dict['target_domain'])

        print(f"source_domain: {para_dict['source_domain']}, its_dict: {src_dict}")
        print(f"target_domain: {para_dict['target_domain']}, its_dict: {tag_dict}")

        src_best_radius_list = best_radius_list(src_dict, para_dict['delta_diff'])
        tag_best_radius_list = best_radius_list(tag_dict, para_dict['delta_diff'])

        radius_a = src_best_radius_list[0]
        radius_b = tag_best_radius_list[0]
        np.savez_compressed(os.path.join(radius_path, para_dict['source_domain']), radius=radius_a)
        np.savez_compressed(os.path.join(radius_path, para_dict['target_domain']), radius=radius_b)
    elif para_dict['radius_assigned']:
        assert para_dict['radius_stats'] is False, 'radius_stats and radius_assigned are contradictory'
        radius_a = int(para_dict['radius_assigned_value'])
        radius_b = int(para_dict['radius_assigned_value'])
    else:
        radius_a = np.load(os.path.join(radius_path, para_dict['source_domain'])+'.npz')['msl']
        radius_b = np.load(os.path.join(radius_path, para_dict['target_domain'])+'.npz')['msl']
        print(f"{para_dict['source_domain']} radius: {radius_a}")
        print(f"{para_dict['target_domain']} radius: {radius_b}")
    
    kaid_model_path = os.path.join('kaidae', para_dict['dataset'])
    create_folders(kaid_model_path)

    if para_dict['train'] is False and para_dict['validation'] is False:
        raise ValueError('train or validation need to be done') 
    
    if para_dict['train']:
        if para_dict['resume']:
            if para_dict['load_latest']:
                # load model
                load_model(model=kaid_ae, file_path=kaid_model_path, description='{}_{}_{}'.format(
                    para_dict['source_domain'], para_dict['target_domain'], 'latest'))
            else:
                load_model(model=kaid_ae, file_path=kaid_model_path, description='{}_{}_{}'.format(
                    para_dict['source_domain'], para_dict['target_domain'], str(para_dict['assigned-epoch'])))


    