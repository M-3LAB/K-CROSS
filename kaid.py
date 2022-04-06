from tkinter.messagebox import NO
import torch
import yaml
import os
import numpy as np
from skimage.util import random_noise
from torch.utils.data import DataLoader
from data_io.ixi import IXI
from data_io.brats import BraTS2021
from tools.utilize import *

from configuration.kaid.config import parse_arguments_kaid
from loss_function.kaid.distance import l1_diff, l2_diff, cosine_similiarity, freq_distance
from model.kaid.complex_nn.fourier_transform import * 
from model.kaid.ae.kaid_ae import Unet 
from model.kaid.ae.complex_ae import ComplexUnet
from model.kaid.complex_nn.fourier_convolve import * 
from loss_function.kaid.focal_freq import FocalFreqLoss, euclidean_freq_loss
from loss_function.kaid.cosine_sim import cosine_sim_loss, complex_cosine_sim_loss
from data_io.nirps import NIRPS

import warnings
warnings.filterwarnings("ignore")



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

    # dataset IO
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
        
        # make sure normal and nosiy loader release the same order of dataset
        normal_loader = DataLoader(ixi_normal_dataset, num_workers=para_dict['num_workers'],
                                   batch_size=para_dict['batch_size'], shuffle=False)

    elif para_dict['dataset'] == 'brats2021':
        assert para_dict['source_domain'] in ['t1', 't2', 'flair']
        assert para_dict['target_domain'] in ['t1', 't2', 'flair']
        brats_normal_dataset = BraTS2021(root=para_dict['data_path'],
                                         modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                         extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                         noise_type='normal',
                                         learn_mode='train', # train or test is meaningless if dataset_spilited is false
                                         transform_data=normal_transform,
                                         data_mode='paired',
                                         data_num=para_dict['data_num'])

        # make sure normal and nosiy loader release the same order of dataset
        normal_loader = DataLoader(brats_normal_dataset, num_workers=para_dict['num_workers'],
                                   batch_size=para_dict['batch_size'], shuffle=False)
        
    else:
        raise NotImplementedError("New Data Has Not Been Implemented")

    batch_limit = int(para_dict['data_num'] / para_dict['batch_size'])

    # debug mode
    if para_dict['debug']:
        para_dict['num_epochs'] = 2
        batch_limit = 2

    # model
    complex_unet = ComplexUnet().to(device)
    unet = Unet().to(device)
    
    # loss
    criterion_freq = FocalFreqLoss(loss_weight=100.0, alpha=1.0, log_matrix=False,
                                   avg_spectrum=False, batch_matrix=False).to(device) 

    criterion_recon = torch.nn.L1Loss().to(device)

    # optimizer
    optimizer_complex = torch.optim.Adam(complex_unet.parameters(), lr=para_dict['lr'],
                                 betas=[para_dict['beta1'], para_dict['beta2']])
    optimizer_normal = torch.optim.Adam(unet.parameters(), lr=para_dict['lr'],
                                 betas=[para_dict['beta1'], para_dict['beta2']])

    checkpoint_path = os.path.join('kaid_ck', para_dict['dataset'], '{}_{}'.format(para_dict['source_domain'], para_dict['target_domain'])) 
    create_folders(tag_path=checkpoint_path)

    # training
    if para_dict['train']:
        for epoch in range(para_dict['num_epochs']):
            for i, batch in enumerate(normal_loader): 
                if i >= batch_limit:
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
                        real_a_noise = torch.tensor(random_noise(real_a.cpu(), mode='gaussian', 
                                                                mean=para_dict['mu'], 
                                                                var=para_dict['sigma'], clip=True)).float().to(device) 

                        real_b_noise = torch.tensor(random_noise(real_b.cpu(), mode='gaussian', 
                                                                mean=para_dict['mu'], 
                                                                var=para_dict['sigma'], clip=True)).float().to(device) 

                        real_a_noise_hat, real_a_noise_z = unet(real_a_noise) 
                        real_b_noise_hat, real_b_noise_z = unet(real_b_noise)

                        real_a_noise_recon_loss = criterion_recon(real_a_noise_hat, real_a_noise)
                        real_b_noise_recon_loss = criterion_recon(real_b_noise_hat, real_b_noise)

                        noisy_recon_loss = real_a_noise_recon_loss + real_b_noise_recon_loss
                        
                        real_a_sim_loss = cosine_sim_loss(real_z=real_a_z, noise_z=real_a_noise_z) 
                        real_b_sim_loss = cosine_sim_loss(real_z=real_b_z, noise_z=real_b_noise_z) 

                        sim_loss = real_a_sim_loss + real_b_sim_loss

                    loss_total = recon_loss + focal_freq_loss

                    if para_dict['noisy_loss']:
                        loss_total = loss_total + noisy_recon_loss + sim_loss

                    optimizer_normal.zero_grad()
                    loss_total.backward()
                    optimizer_normal.step()

                    infor = '\r{}[Epoch {} / {}] [Batch {}/{}] [Recon Loss: {:.4f}] [Focal Freq Loss: {:.4f}]'.format(
                                '', epoch+1, para_dict['num_epochs'], i+1, batch_limit, recon_loss.item(), focal_freq_loss.item())

                    if para_dict['noisy_loss']:
                        infor = '{} [Noisy Recon Loss: {:.4f}] [Sim Loss: {:.4f}]'.format(infor, noisy_recon_loss, sim_loss)

                    print(infor, flush=True, end='  ')    

                elif para_dict['method'] == 'complex':
                    real_a_freq = torch_fft(real_a, normalized_method='ortho')
                    real_b_freq = torch_fft(real_b, normalized_method='ortho')

                    real_a_freq_hat, real_a_freq_z = complex_unet(real_a_freq)
                    real_b_freq_hat, real_b_freq_z = complex_unet(real_b_freq)

                    real_a_freq_loss = euclidean_freq_loss(real_freq=real_a_freq, recon_freq=real_a_freq_hat)
                    real_b_freq_loss = euclidean_freq_loss(real_freq=real_b_freq, recon_freq=real_b_freq_hat)

                    freq_loss = real_a_freq_loss + real_b_freq_loss

                    loss_total = freq_loss

                    if para_dict['noisy_loss']:
                        real_a_noise = torch.tensor(random_noise(real_a.cpu(), mode='gaussian', 
                                                                mean=para_dict['mu'], 
                                                                var=para_dict['sigma'], clip=True)).float().to(device) 

                        real_b_noise = torch.tensor(random_noise(real_b.cpu(), mode='gaussian', 
                                                                mean=para_dict['mu'], 
                                                                var=para_dict['sigma'], clip=True)).float().to(device)

                        real_a_noise_freq = torch_fft(real_a_noise, normalized_method='ortho')
                        real_b_noise_freq = torch_fft(real_b_noise, normalized_method='ortho')

                        real_a_noise_freq_hat, real_a_noise_freq_z = complex_unet(real_a_noise_freq)
                        real_b_noise_freq_hat, real_b_noise_freq_z = complex_unet(real_b_noise_freq)

                        real_a_noise_freq_loss = euclidean_freq_loss(real_freq=real_a_noise_freq, recon_freq=real_a_noise_freq_hat)
                        real_b_noise_freq_loss = euclidean_freq_loss(real_freq=real_b_noise_freq, recon_freq=real_b_noise_freq_hat)

                        noise_freq_loss = real_a_noise_freq_loss + real_b_noise_freq_loss

                        real_a_freq_sim_loss = complex_cosine_sim_loss(real_freq_z=real_a_freq_z, noise_freq_z=real_a_noise_freq_z)
                        real_b_freq_sim_loss = complex_cosine_sim_loss(real_freq_z=real_b_freq_z, noise_freq_z=real_b_noise_freq_z)
                        sim_freq_loss = real_a_freq_sim_loss + real_b_freq_sim_loss

                        loss_total = loss_total + noise_freq_loss + sim_freq_loss
                        
                    optimizer_complex.zero_grad()
                    loss_total.backward()
                    optimizer_complex.step()

                    infor = '\r{}[Epoch {} / {}] [Batch {}/{}] [Freq Loss: {:.4f}]'.format(
                                '', epoch+1, para_dict['num_epochs'], i+1, batch_limit, freq_loss.item())
                    if para_dict['noisy_loss']:
                        infor= '{} [Noise Freq Loss: {:.4f}] [Sim Loss: {:.4f}]'.format(infor, noise_freq_loss.item(), sim_freq_loss.item())

                    print(infor, flush=True, end='  ')    

                elif para_dict['method'] == 'combined':
                    real_a_hat, real_a_z = unet(real_a)
                    real_b_hat, real_b_z = unet(real_b)

                    real_a_recon_loss = criterion_recon(real_a_hat, real_a) 
                    real_b_recon_loss = criterion_recon(real_b_hat, real_b) 
                    recon_loss = real_a_recon_loss + real_b_recon_loss

                    real_a_freq = torch_fft(real_a, normalized_method='ortho')
                    real_b_freq = torch_fft(real_b, normalized_method='ortho')

                    real_a_freq_hat, real_a_freq_z = complex_unet(real_a_freq)
                    real_b_freq_hat, real_b_freq_z = complex_unet(real_b_freq)

                    real_a_freq_loss = euclidean_freq_loss(real_freq=real_a_freq, recon_freq=real_a_freq_hat)
                    real_b_freq_loss = euclidean_freq_loss(real_freq=real_b_freq, recon_freq=real_b_freq_hat)
                    freq_loss = real_a_freq_loss + real_b_freq_loss

                    loss_total = recon_loss + freq_loss

                    if para_dict['noisy_loss']:
                        real_a_noise = torch.tensor(random_noise(real_a.cpu(), mode='gaussian', 
                                                                mean=para_dict['mu'], 
                                                                var=para_dict['sigma'], clip=True)).float().to(device) 

                        real_b_noise = torch.tensor(random_noise(real_b.cpu(), mode='gaussian', 
                                                                mean=para_dict['mu'], 
                                                                var=para_dict['sigma'], clip=True)).float().to(device) 

                        real_a_noise_hat, real_a_noise_z = unet(real_a_noise) 
                        real_b_noise_hat, real_b_noise_z = unet(real_b_noise)

                        real_a_noise_recon_loss = criterion_recon(real_a_noise_hat, real_a_noise)
                        real_b_noise_recon_loss = criterion_recon(real_b_noise_hat, real_b_noise)

                        noisy_recon_loss = real_a_noise_recon_loss + real_b_noise_recon_loss
                        
                        real_a_sim_loss = cosine_sim_loss(real_z=real_a_z, noise_z=real_a_noise_z) 
                        real_b_sim_loss = cosine_sim_loss(real_z=real_b_z, noise_z=real_b_noise_z) 

                        sim_loss = real_a_sim_loss + real_b_sim_loss

                        real_a_noise_freq = torch_fft(real_a_noise, normalized_method='ortho')
                        real_b_noise_freq = torch_fft(real_b_noise, normalized_method='ortho')

                        real_a_noise_freq_hat, real_a_noise_freq_z = complex_unet(real_a_noise_freq)
                        real_b_noise_freq_hat, real_b_noise_freq_z = complex_unet(real_b_noise_freq)

                        real_a_noise_freq_loss = euclidean_freq_loss(real_freq=real_a_noise_freq, recon_freq=real_a_noise_freq_hat)
                        real_b_noise_freq_loss = euclidean_freq_loss(real_freq=real_b_noise_freq, recon_freq=real_b_noise_freq_hat)

                        noise_freq_loss = real_a_noise_freq_loss + real_b_noise_freq_loss

                        real_a_freq_sim_loss = complex_cosine_sim_loss(real_freq_z=real_a_freq_z, noise_freq_z=real_a_noise_freq_z)
                        real_b_freq_sim_loss = complex_cosine_sim_loss(real_freq_z=real_b_freq_z, noise_freq_z=real_b_noise_freq_z)
                        sim_freq_loss = real_a_freq_sim_loss + real_b_freq_sim_loss
                    
                        loss_total = loss_total + noisy_recon_loss + sim_loss + noise_freq_loss + sim_freq_loss
 

                    optimizer_complex.zero_grad()
                    optimizer_normal.zero_grad()

                    loss_total.backward()

                    optimizer_complex.step()
                    optimizer_normal.step()

                    infor = '\r{}[Epoch {} / {}] [Batch {}/{}] [Recon Loss: {:.4f}] [Freq Loss: {:.4f}]'.format(
                                '', epoch+1, para_dict['num_epochs'], i+1, batch_limit, recon_loss.item(), freq_loss.item())

                    if para_dict['noisy_loss']:
                        infor = '{} [Noisy Recon Loss: {:.4f}] [Sim Loss: {:.4f}] [Noisy Freq Loss: {:.4f}] [Sim Freq Loss: {:.4f}]'.format(
                            infor, noisy_recon_loss.item(), sim_loss.item(), noise_freq_loss.item(), sim_freq_loss.item() 
                        )

                    print(infor, flush=True, end='  ')
                else:
                    raise NotImplementedError('The method has not been implemented yet')

            # save model 
            if para_dict['method'] == 'normal':
                if para_dict['noisy_loss']:
                    save_model(model=unet, file_path=checkpoint_path, infor='normal_noisy', save_previous=True)
                else:
                    save_model(model=unet, file_path=checkpoint_path, infor='normal', save_previous=True)

            elif para_dict['method'] == 'complex':
                if para_dict['noisy_loss']:
                    save_model(model=complex_unet, file_path=checkpoint_path, infor='complex_noisy', save_previous=True)
                else:
                    save_model(model=complex_unet, file_path=checkpoint_path, infor='complex', save_previous=True)

            elif para_dict['method'] == 'combined':
                if para_dict['noisy_loss']:
                    save_model(model=complex_unet, file_path=checkpoint_path, infor='combined_complex_noisy', save_previous=True)
                    save_model(model=unet, file_path=checkpoint_path, infor='combined_normal_noisy', save_previous=True)
                else:
                    save_model(model=complex_unet, file_path=checkpoint_path, infor='combined_complex', save_previous=True)
                    save_model(model=unet, file_path=checkpoint_path, infor='combined_normal', save_previous=True)

    # inference
    if para_dict['validate']: 
        nirps_path = para_dict['nirps_path']
        if para_dict['infer_range'] == 'all':
            regions = ['ixi', 'brats2021']
            modalities = {'ixi': ['t2', 'pd'],
                        'brats2021': ['t1', 't2', 'flair']}
        elif para_dict['infer_range'] == 'ixi':
                regions = ['ixi']
                modalities = {'ixi': ['t2', 'pd']}
        elif para_dict['infer_range'] == 'brats2021':
                regions = ['brats2021']
                modalities = {'brats2021': ['t1', 't2', 'flair']}
        else:
            raise NotImplementedError


        models = ['cyclegan'] 
        epochs = [i for i in range(1, 51)]

        nirps_dataset = NIRPS(nirps_path=nirps_path, regions=regions, modalities=modalities, models=models, epochs=epochs)
        nirps_loader = DataLoader(nirps_dataset, batch_size=1, num_workers=1, shuffle=False)
        print('load nirps dataset, size: {}'.format(len(nirps_dataset)))

        # load models
        if para_dict['method'] == 'normal':
            if para_dict['noisy_loss']:
                unet = load_model(model=unet, file_path=checkpoint_path, description='normal_noisy')
            else:
                unet = load_model(model=unet, file_path=checkpoint_path, description='normal')
            
        elif para_dict['method'] == 'complex':
            if para_dict['noisy_loss']:
                complex_unet = load_model(model=complex_unet, file_path=checkpoint_path, description='complex_noisy')
            else:
                complex_unet = load_model(model=complex_unet, file_path=checkpoint_path, description='complex')
            
        elif para_dict['method'] == 'combined':
            if para_dict['noisy_loss']:
                unet = load_model(model=unet, file_path=checkpoint_path, description='combined_normal_noisy')
                complex_unet = load_model(model=complex_unet, file_path=checkpoint_path, description='combined_complex_noisy')
            else:
                unet = load_model(model=unet, file_path=checkpoint_path, description='combined_normal')
                complex_unet = load_model(model=complex_unet, file_path=checkpoint_path, description='combined_complex')

        # score images of nirps dataset
        arti_values = []
        kaid_values = []
        mae_values = []
        ssim_values = []
        psnr_values = []
        for batch in nirps_loader:
            img = batch['img'].float().to(device)
            gt = batch['gt'].float().to(device)
            name = batch['name'][0]

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

                if para_dict['noisy_loss']:
                    save_metric_result(result=kaid, file_path=name, description='kaid_normal_noisy')
                else:
                    save_metric_result(result=kaid, file_path=name, description='kaid_normal')

                kaid_values.append(kaid)

            elif para_dict['method'] == 'complex':
                img_freq = torch_fft(img, normalized_method='ortho')
                gt_freq = torch_fft(gt, normalized_method='ortho')

                img_freq_z = complex_unet.encode(img_freq)
                gt_freq_z = complex_unet.encode(gt_freq)

                kaid = freq_distance(real_z=gt_freq_z, fake_z=img_freq_z).item()

                if para_dict['noisy_loss']:
                    save_metric_result(result=kaid, file_path=name, description='kaid_complex_noisy')
                else:
                    save_metric_result(result=kaid, file_path=name, description='kaid_complex')

                kaid_values.append(kaid)

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

                if para_dict['noisy_loss']:
                    save_metric_result(result=kaid, file_path=name, description='kaid_combined_noisy')
                else:
                    save_metric_result(result=kaid, file_path=name, description='kaid_combined')

                kaid_values.append(kaid)
             
            else:
                raise NotImplementedError

            arti = load_metric_result(name, 'artificial')
            mae = load_metric_result(name, 'mae')
            psnr = load_metric_result(name, 'psnr')
            ssim = load_metric_result(name, 'ssim')

            arti_values.append(arti)
            mae_values.append(mae)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

        # calculate metric consistency
        kaid_values = uniform_result(kaid_values, reverse=True)
        mae_values = uniform_result(mae_values, reverse=True)
        psnr_values = uniform_result(psnr_values, reverse=False)
        ssim_values = uniform_result(ssim_values, reverse=False)

        kaid_consistency = calculate_metric_consistency(kaid_values, arti_values) 
        mae_consistency = calculate_metric_consistency(mae_values, arti_values) 
        psnr_consistency = calculate_metric_consistency(psnr_values, arti_values)
        ssim_consistency = calculate_metric_consistency(ssim_values, arti_values) 


        infor = '[Epoch {}/{}] kaid: {:.4f} mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
            1, para_dict['num_epochs'], kaid_consistency, mae_consistency, psnr_consistency, ssim_consistency)
        print(infor)

        save_log(infor, file_path, description='metric_result')
                


                


    