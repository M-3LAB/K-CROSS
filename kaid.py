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
from metrics.kaid.stats import mask_stats, best_msl_list 
from model.kaid.ae.kaid_ae import KAIDAE


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
                                 data_num=para_dict['pair_num'],
                                 dataset_splited=False)
        
        ixi_noise_dataset = IXI(root=para_dict['data_path'],
                                modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                noise_type=para_dict['noise_type'],
                                learn_mode='train', #train or test is meaningless if dataset_splited is false
                                transform_data=noise_transform,
                                data_mode='paired',
                                data_num=para_dict['pair_num'],
                                dataset_splited=False)

        #TODO: make sure normal and nosiy loader release the same order of dataset
        normal_loader = DataLoader(ixi_normal_dataset, num_workers=para_dict['num_workers'],
                                   batch_size=para_dict['batch_size'], shuffle=False)

        noisy_loader = DataLoader(ixi_noise_dataset, num_workers=para_dict['num_workers'],
                                  batch_size=para_dict['batch_size'], shuffle=False)

        test_loader = DataLoader(ixi_normal_dataset, num_workers=para_dict['num_workers'],
                                 batch_size=1, shuffle=False)

        
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
                                         data_num=para_dict['pair_num'])

        brats_noise_dataset = BraTS2021(root=para_dict['data_path'],
                                        modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                        noise_type=para_dict['noise_type'],
                                        learn_mode='train',
                                        extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                        transform_data=noise_transform,
                                        data_mode='paired',
                                        data_num=para_dict['pair_num'])
        
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
        batch_limit = 2
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
    msl_path = os.path.join(para_dict['msl_path'], para_dict['dataset']) 
    create_folders(msl_path) 

    if para_dict['msl_stats']:
        assert para_dict['msl_assigned'] is False, 'msl_stats and msl_assigned are contradictory'
        src_dict, tag_dict = mask_stats(normal_loader, para_dict['source_domain'], 
                                        para_dict['target_domain'])

        print(f"source_domain: {para_dict['source_domain']}, its_dict: {src_dict}")
        print(f"target_domain: {para_dict['target_domain']}, its_dict: {tag_dict}")

        src_best_msl_list = best_msl_list(src_dict, para_dict['delta_diff'])
        tag_best_msl_list = best_msl_list(tag_dict, para_dict['delta_diff'])

        msl_a = src_best_msl_list[0]
        msl_b = tag_best_msl_list[0]
        np.savez_compressed(os.path.join(msl_path, para_dict['source_domain']), msl=msl_a)
        np.savez_compressed(os.path.join(msl_path, para_dict['target_domain']), msl=msl_b)
    elif para_dict['msl_assigned']:
        assert para_dict['msl_stats'] is False, 'msl_stats and msl_assigned are contradictory'
        msl_a = int(para_dict['msl_assigned_value'])
        msl_b = int(para_dict['msl_assigned_value'])
    else:
        msl_a = np.load(os.path.join(msl_path, para_dict['source_domain'])+'.npz')['msl']
        msl_b = np.load(os.path.join(msl_path, para_dict['target_domain'])+'.npz')['msl']
        print(f"{para_dict['source_domain']} msl: {msl_a}")
        print(f"{para_dict['target_domain']} msl: {msl_b}")
    
    kaid_model_path = os.path.join('kaid', 'model')

    if para_dict['train'] is False and para_dict['validation'] is False:
        raise ValueError('train or validation need to be done')
    # Training 
    #TODO: Alternative Training for different training loader
    if para_dict['train']:
        if para_dict['resume']:
            # load model
            load_model(model=kaid_ae, file_path=kaid_model_path, description='{}/{}_{}'.format(
                para_dict['dataset'], para_dict['source_domain'], para_dict['target_domain']))
        
        for epoch in range(para_dict['num_epochs']):
            for i, batch in enumerate(normal_loader): 
            #TODO: noisy loader

                if i > batch_limit:
                    break

                real_a = batch[para_dict['source_domain']]
                real_b = batch[para_dict['target_domain']]

                # Fourier Transform 
                real_a_kspace = torch_fft(real_a)
                real_b_kspace = torch_fft(real_b)

                real_a_hf = torch_high_pass_filter(real_a_kspace, msl_a)
                real_b_hf = torch_high_pass_filter(real_b_kspace, msl_b)

                real_a_lf = torch_low_pass_filter(real_a_kspace, msl_a)
                real_b_lf = torch_low_pass_filter(real_b_kspace, msl_b)

                """
                Magnitude: sqrt(re^2 + im^2) tells you the amplitude of the component at the corresponding frequency
                Phase: atan2(im, re) tells you the relative phase of that component
                """

                real_a_hf_mag = torch.abs(real_a_hf).to(device)
                real_a_lf_mag = torch.abs(real_a_lf).to(device)

                real_b_hf_mag = torch.abs(real_b_hf).to(device)
                real_b_lf_mag = torch.abs(real_b_lf).to(device)

                optimizer.zero_grad()

                real_a_hf_z, real_a_hf_hat = kaid_ae(real_a_hf_mag)
                real_a_lf_z, real_a_lf_hat = kaid_ae(real_a_lf_mag)

                real_b_hf_z, real_b_hf_hat = kaid_ae(real_b_hf_mag)
                real_b_lf_z, real_b_lf_hat = kaid_ae(real_b_lf_mag)

                """
                Reconstruction
                """
                loss_recon_real_a_hf = criterion_recon(real_a_hf_mag, real_a_hf_hat) 
                loss_recon_real_b_hf = criterion_recon(real_b_hf_mag, real_b_hf_hat)

                loss_recon_real_a_lf = criterion_recon(real_a_lf_mag, real_a_lf_hat) 
                loss_recon_real_b_lf = criterion_recon(real_b_lf_mag, real_b_lf_hat)

                loss_recon = (loss_recon_real_a_hf + loss_recon_real_b_hf 
                                    + loss_recon_real_a_lf + loss_recon_real_b_lf)

                """
                Contrastive Loss
                """
                loss_high_frequency = criterion_high_freq(real_a_hf_z, real_b_hf_z) 
                loss_low_frequency = criterion_low_freq(real_a_lf_z, real_b_lf_z)
                contrastive_loss = (para_dict['lambda_hf'] * loss_high_frequency -
                                    para_dict['lambda_lf'] * loss_low_frequency)

                loss_total = (para_dict['lambda_contrastive'] * contrastive_loss + 
                              para_dict['lambda_recon'] * loss_recon)

                loss_total.backward()
                optimizer.step()
                lr_scheduler.step()

                # Print Log
                infor = '\r{}[Batch {}/{}] [Recons loss: {:.4f}] [contrastive loss: {:.4f}]'.format(
                            '', i, batch_limit, loss_recon.item(), contrastive_loss.item())

                print(infor)         

        
        save_model(model=kaid_ae, file_path='{}/checkpoint'.format(kaid_model_path), infor='{}/{}_{}'.format(
                   para_dict['dataset'], para_dict['source_domain'], para_dict['target_domain'])) 
    
    if para_dict['validate']:
        load_model(model=kaid_ae, file_path=kaid_model_path, description='{}/{}_{}'.format(para_dict['dataset'], para_dict['source_domain'], para_dict['target_domain']))

        # Score Prediction
        #TODO: Load GAN Model and KAID  
        if para_dict['test_model'] == 'cyclegan':
            generator_from_a_to_b = CycleGen().to(device) 
            generator_from_b_to_a = CycleGen().to(device)

        elif para_dict['test_model'] == 'munit':
            encoder_from_a_to_b = MUE().to(device)
            decoder_from_a_to_b = MUD().to(device)
            encoder_from_b_to_a = MUE().to(device)
            decoder_from_b_to_a = MUD().to(device)

        elif para_dict['test_model'] == 'unit':
            encoder_from_a_to_b = UE().to(device)
            generator_from_a_to_b = UG().to(device) 
            encoder_from_b_to_a = UE().to(device)
            generator_from_b_to_a = UG().to(device) 

        else:
            raise NotImplementedError('GAN Model Has Not Been Implemented Yet')
    
        #TODO: synthesis data loader
        # Single Image Quality, batchsize=1
        for i, batch in enumerate(test_loader): 
            if i > batch_limit:
                break

            real_a = batch[para_dict['source_domain']].to(device)
            real_b = batch[para_dict['target_domain']].to(device)
            
            # Synthesize Image 
            if para_dict['test_model'] == 'cyclegan':
                fake_b = generator_from_a_to_b(real_a).to(device)
                fake_a = generator_from_b_to_a(real_b).to(device)
            elif para_dict['test_model'] == 'munit':
                pass
            elif para_dict['test_model'] == 'unit':
                pass
            else:
                raise NotImplementedError('Synthesis Model Not Implemented Yet')
            
            #Distance 
            real_a_z = kaid_ae.encode(real_a)    
            fake_a_z = kaid_ae.encode(fake_a)

            real_b_z = kaid_ae.encode(real_b)
            fake_b_z = kaid_ae.encode(fake_b)

            if para_dict['diff_method'] == 'l1':
                diff_a = l1_diff(real_a_z, fake_a_z)
                diff_b = l1_diff(real_b_z, fake_b_z)
            elif para_dict['diff_method'] == 'l2':
                diff_a = l2_diff(real_a_z, fake_a_z)
                diff_b = l2_diff(real_b_z, fake_b_z)
            elif para_dict['diff_method'] == 'cos':
                diff_a = cosine_similiarity(real_a_z, fake_a_z)
                diff_b = cosine_similiarity(real_b_z, fake_b_z)
            else:
                raise NotImplementedError('The Difference Method Has Not Been Calculated Yet')

            print(f"The mean diff of Modality {para_dict['source_domain']} : {torch.mean(diff_a)}")
            print(f"The mean diff of Modality {para_dict['target_domain']} : {torch.mean(diff_b)}")


        #TODO: Comparision on NIRPS 
    