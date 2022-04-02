from ast import Not
from locale import normalize
from tools.visualize import np_scaling_kspace, torch_scaling_kspace
import torch
import yaml
import os
import numpy as np
from skimage.util import random_noise
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
from loss_function.kaid.distance import l1_diff, l2_diff, cosine_similiarity, freq_distance
from model.kaid.complex_nn.fourier_transform import * 
from model.kaid.complex_nn.power_spectrum import *
from metrics.kaid.stats import mask_stats, best_radius_list 
from model.kaid.ae.kaid_ae import Unet 
from model.kaid.ae.complex_ae import ComplexUnet
from model.kaid.complex_nn.fourier_convolve import * 
from loss_function.kaid.focal_freq import FocalFreqLoss, euclidean_freq_loss
from data_io.nirps import NIRPS

from tools.visualize import *
#import matplotlib.pyplot as plt

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
        noise_transform = [{'mu':para_dict['mu'], 'sigma':para_dict['sigma'],
                            'size':(para_dict['size'], para_dict['size'])},
                           {'mu':para_dict['mu'], 'sigma':para_dict['sigma'],
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
        
        #ixi_noise_dataset = IXI(root=para_dict['data_path'],
        #                        modalities=[para_dict['source_domain'], para_dict['target_domain']],
        #                        extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
        #                        noise_type=para_dict['noise_type'],
        #                        learn_mode='train', #train or test is meaningless if dataset_splited is false
        #                        transform_data=noise_transform,
        #                        data_mode='paired',
        #                        data_num=para_dict['data_num'],
        #                        dataset_splited=False)
        
        #TODO: make sure normal and nosiy loader release the same order of dataset
        normal_loader = DataLoader(ixi_normal_dataset, num_workers=para_dict['num_workers'],
                                   batch_size=para_dict['batch_size'], shuffle=False)

        #noisy_loader = DataLoader(ixi_noise_dataset, num_workers=para_dict['num_workers'],
        #                          batch_size=para_dict['batch_size'], shuffle=False)

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

        #brats_noise_dataset = BraTS2021(root=para_dict['data_path'],
        #                                modalities=[para_dict['source_domain'], para_dict['target_domain']],
        #                                noise_type=para_dict['noise_type'],
        #                                learn_mode='train',
        #                                extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
        #                                transform_data=noise_transform,
        #                                data_mode='paired',
        #                                data_num=para_dict['data_num'])
        
        
        #TODO: make sure normal and nosiy loader release the same order of dataset
        normal_loader = DataLoader(brats_normal_dataset, num_workers=para_dict['num_workers'],
                                   batch_size=para_dict['batch_size'], shuffle=False)

        #noisy_loader = DataLoader(brats_noise_dataset, num_workers=para_dict['num_workers'],
        #                          batch_size=para_dict['batch_size'], shuffle=False)
        
    else:
        raise NotImplementedError("New Data Has Not Been Implemented")

    # Debug Mode
    if para_dict['debug']:
        batch_limit = 1
    else:
        batch_limit = int(para_dict['data_num'] / para_dict['batch_size'])

    # Model
    complex_unet = ComplexUnet().to(device)
    unet = Unet().to(device)
    
    # Loss
    #TODO: Add Focal Freq Loss
    criterion_freq = FocalFreqLoss(loss_weight=100.0, alpha=1.0, log_matrix=False,
                                   avg_spectrum=False, batch_matrix=False).to(device) 

    criterion_recon = torch.nn.L1Loss().to(device)

    # Optimizer
    optimizer_complex = torch.optim.Adam(complex_unet.parameters(), lr=para_dict['lr'],
                                 betas=[para_dict['beta1'], para_dict['beta2']])
    optimizer_normal = torch.optim.Adam(unet.parameters(), lr=para_dict['lr'],
                                 betas=[para_dict['beta1'], para_dict['beta2']])

    checkpoint_path = os.path.join('kaid_ck', para_dict['dataset']) 
    create_folders(tag_path=checkpoint_path)

    if para_dict['train']:
        for epoch in range(para_dict['num_epochs']):
            for i, batch in enumerate(normal_loader): 
                if i > batch_limit:
                    break
                
                real_a = batch[para_dict['source_domain']].to(device)
                real_b = batch[para_dict['target_domain']].to(device)

                if para_dict['method'] == 'normal':
                    real_a_hat, real_a_z = unet(real_a)
                    real_b_hat, real_b_z = unet(real_b)

                    real_a_recon_loss = criterion_recon(real_a_hat, real_a) 
                    real_b_recon_loss = criterion_recon(real_b_hat, real_b) 
                    recon_loss = real_a_recon_loss + real_b_recon_loss

                    real_a_freq_loss = criterion_freq(real_a_hat, real_a)
                    real_b_freq_loss = criterion_freq(real_b_hat, real_b)
                    focal_freq_loss = real_a_freq_loss + real_b_freq_loss

                    if para_dict['noisy_loss']:
                        real_a_noise = torch.tensor(random_noise(real_a, mode='gaussian', 
                                                                mean=para_dict['mu'], 
                                                                var=para_dict['sigma'], clip=True)).to(device) 

                        real_b_noise = torch.tensor(random_noise(real_b, mode='gaussian', 
                                                                mean=para_dict['mu'], 
                                                                var=para_dict['sigma'], clip=True)).to(device) 


                    loss_total = recon_loss + focal_freq_loss

                    optimizer_normal.zero_grad()
                    loss_total.backward()
                    optimizer_normal.step()

                    infor = '\r{}[Batch {}/{}] [Recon Loss: {:.4f}] [Focal Freq Loss: {:.4f}]'.format(
                                '', i+1, batch_limit, recon_loss.item(), focal_freq_loss.item())

                    print(infor, flush=True, end='  ')    

                elif para_dict['method'] == 'complex':
                    real_a_freq = torch_fft(real_a, normalized_method='ortho')
                    real_b_freq = torch_fft(real_b, normalized_method='ortho')

                    real_a_freq_hat, _ = complex_unet(real_a_freq)
                    real_b_freq_hat, _ = complex_unet(real_b_freq)

                    real_a_freq_loss = euclidean_freq_loss(real_freq=real_a_freq, recon_freq=real_a_freq_hat)
                    real_b_freq_loss = euclidean_freq_loss(real_freq=real_b_freq, recon_freq=real_b_freq_hat)

                    freq_loss = real_a_freq_loss + real_b_freq_loss

                    loss_total = freq_loss

                    optimizer_complex.zero_grad()
                    loss_total.backward()
                    optimizer_complex.step()

                    infor = '\r{}[Batch {}/{}] [Freq Loss: {:.4f}]'.format(
                                '', i+1, batch_limit, freq_loss.item())

                    print(infor, flush=True, end='  ')    

                elif para_dict['method'] == 'combined':
                    real_a_hat, _ = unet(real_a)
                    real_b_hat, _ = unet(real_b)

                    real_a_recon_loss = criterion_recon(real_a_hat, real_a) 
                    real_b_recon_loss = criterion_recon(real_b_hat, real_b) 
                    recon_loss = real_a_recon_loss + real_b_recon_loss

                    real_a_freq = torch_fft(real_a, normalized_method='ortho')
                    real_b_freq = torch_fft(real_b, normalized_method='ortho')

                    real_a_freq_hat, _ = complex_unet(real_a_freq)
                    real_b_freq_hat, _ = complex_unet(real_b_freq)

                    real_a_freq_loss = euclidean_freq_loss(real_freq=real_a_freq, recon_freq=real_a_freq_hat)
                    real_b_freq_loss = euclidean_freq_loss(real_freq=real_b_freq, recon_freq=real_b_freq_hat)
                    freq_loss = real_a_freq_loss + real_b_freq_loss

                    loss_total = recon_loss + freq_loss

                    optimizer_complex.zero_grad()
                    optimizer_normal.zero_grad()

                    loss_total.backward()

                    optimizer_complex.step()
                    optimizer_normal.step()

                    infor = '\r{}[Batch {}/{}] [Recon Loss: {:.4f}] [Freq Loss: {:.4f}]'.format(
                                '', i+1, batch_limit, recon_loss.item(), freq_loss.item())

                    print(infor, flush=True, end='  ')
                else:
                    raise NotImplementedError('The method has not been implemented yet')
        
        if para_dict['method'] == 'normal':
            save_model(model=unet, file_path=checkpoint_path, infor='normal', save_previous=True)
        elif para_dict['method'] == 'complex':
            save_model(model=complex_unet, file_path=checkpoint_path, infor='complex', save_previous=True)
        elif para_dict['method'] == 'combined':
            save_model(model=complex_unet, file_path=checkpoint_path, infor='combined_complex', save_previous=True)
            save_model(model=unet, file_path=checkpoint_path, infor='combined_normal', save_previous=True)

    if para_dict['validate']: 
        nirps_path = para_dict['nirps_path']
        regions = ['ixi']
        modalities = {'ixi': ['t2', 'pd']}
        models = ['cyclegan'] 
        epochs = [i for i in range(1, 51)]

        nirps_dataset = NIRPS(nirps_path=nirps_path, regions=regions, modalities=modalities, models=models, epochs=epochs)
        nirps_loader = DataLoader(nirps_dataset, batch_size=1, num_workers=1, shuffle=False)
        print('load nirps dataset, size: {}'.format(len(nirps_dataset)))

        #load models
        if para_dict['method'] == 'normal':
            unet = load_model(model=unet, file_path=checkpoint_path, description='normal')
            
        elif para_dict['method'] == 'complex':
            complex_unet = load_model(model=complex_unet, file_path=checkpoint_path, description='complex')
        elif para_dict['method'] == 'combined':
            unet = load_model(model=unet, file_path=checkpoint_path, description='combined_normal')
            complex_unet = load_model(model=complex_unet, file_path=checkpoint_path, description='combined_complex')

        for batch in nirps_loader:

            img = batch['img'].float().to(device)
            gt = batch['gt'].float().to(device)
            name = batch['name']

            if para_dict['method'] == 'normal':
                img_z = unet.encode(img)
                gt_z = unet.encode(gt)

                if para_dict['diff'] == 'l1':
                    kaid = l1_diff(real_z=gt_z, fake_z=img_z).item()
                elif para_dict['diff'] == 'l2':
                    kaid = l2_diff(real_z=gt_z, fake_z=img_z).item()
                elif para_dict['diff'] == 'cos':
                    kaid = cosine_similiarity(real_z=gt_z, fake_z=img_z).item()
                else:
                    raise ValueError

                save_metric_result(result=kaid, file_path=name[0], description='kaid_normal')

            elif para_dict['method'] == 'complex':

                img_freq = torch_fft(img, normalized_method='ortho')
                gt_freq = torch_fft(gt, normalized_method='ortho')

                img_freq_z = complex_unet.encode(img_freq)
                gt_freq_z = complex_unet.encode(gt_freq)

                kaid = freq_distance(real_z=gt_freq_z, fake_z=img_freq_z).item()

                save_metric_result(result=kaid, file_path=name[0], description='kaid_complex')

            elif para_dict['method'] == 'combined':

                img_z = unet.encode(img)
                gt_z = unet.encode(gt)

                img_freq = torch_fft(img, normalized_method='ortho')
                gt_freq = torch_fft(gt, normalized_method='ortho')
                img_freq_z = complex_unet.encode(img_freq)
                gt_freq_z = complex_unet.encode(gt_freq)

                if para_dict['diff'] == 'l1':
                    kaid = (l1_diff(real_z=gt_z, fake_z=img_z) + freq_distance(real_z=gt_freq_z, fake_z=img_freq_z)).item() 
                elif para_dict['diff'] == 'l2':
                    kaid = (l2_diff(real_z=gt_z, fake_z=img_z) + freq_distance(real_z=gt_freq_z, fake_z=img_freq_z)).item() 
                else:
                    raise ValueError

                save_metric_result(result=kaid, file_path=name[0], description='kaid_combined')

            else:
                raise NotImplementedError

    

    
                


                


    