from distutils.command.config import config
from lib2to3.pytree import convert
import torch
import yaml
import os
from tools.utilize import *
from data_io.brats import BraTS2021
from data_io.ixi import IXI
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from centralized.cyclegan import CycleGAN
from centralized.munit import Munit
from centralized.unit import Unit

import warnings
warnings.filterwarnings("ignore")

class CentralizedTrain():
    def __init__(self, args):
        self.args = args

    def load_config(self):
        with open('./configuration/3_dataset_base/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_model = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/2_train_base/centralized_training.yaml', 'r') as f:
            config_train = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/1_model_base/{}.yaml'.format(self.args.model), 'r') as f:
            config_dataset = yaml.load(f, Loader=yaml.SafeLoader)

        config = override_config(config_model, config_train)
        config = override_config(config, config_dataset)
        self.para_dict = merge_config(config, self.args)
        self.args = extract_config(self.args)


    def preliminary(self):
        print('---------------------')
        print(self.args)
        print('---------------------')
        print(self.para_dict)
        print('---------------------')

        seed_everything(self.para_dict['seed'])

        device, device_ids = parse_device_list(self.para_dict['gpu_ids'], int(self.para_dict['gpu_id']))
        self.device = torch.device("cuda", device)

        self.file_path = record_path(self.para_dict)
        if self.para_dict['save_log']:
            save_arg(self.para_dict, self.file_path)
            save_script(__file__, self.file_path)

        # save model
        self.best_psnr_a = 0.
        self.best_psnr_b = 0.
        self.load_model_path = self.para_dict['load_model_dir']

        self.fid_stats_from_a_to_b = '{}/{}/{}.npz'.format(
            self.para_dict['fid_dir'], self.para_dict['dataset'], self.para_dict['target_domain'])
        self.fid_stats_from_b_to_a = '{}/{}/{}.npz'.format(
            self.para_dict['fid_dir'], self.para_dict['dataset'], self.para_dict['source_domain'])

        # check fid stats
        if self.para_dict['fid']:
            if not os.path.exists(self.fid_stats_from_a_to_b) or not os.path.exists(self.fid_stats_from_b_to_a):
                os.system(r'python3 ./fid_stats/gen_fid_stats.py --dataset {} --source-domain {} --target-domain {} --gpu-id {} --valid-path {}'.format(
                    self.para_dict['dataset'], self.para_dict['source_domain'], self.para_dict['target_domain'], self.para_dict['gpu_id'], self.para_dict['valid_path']))

            if not os.path.exists(self.fid_stats_from_a_to_b):
                raise NotImplementedError('FID Still Not be Implemented Yet')
            else:
                print('fid stats from a to b: {}'.format(self.fid_stats_from_a_to_b))

            if not os.path.exists(self.fid_stats_from_b_to_a):
                raise NotImplementedError('FID Still Not be Implemented Yet')
            else:
                print('fid stats from b to a: {}'.format(self.fid_stats_from_b_to_a))

        print('work dir: {}'.format(self.file_path))
        print('noise type: {}'.format(self.para_dict['noise_type']))

        if self.para_dict['noise_type'] == 'reg':
            print('noise level: {}'.format(self.para_dict['noise_level']))


    def load_data(self):
        self.normal_transform = [{'degrees':0, 'translate':[0.00, 0.00],
                                     'scale':[1.00, 1.00], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                 {'degrees':0, 'translate':[0.00, 0.00],
                                  'scale':[1.00, 1.00], 
                                  'size':(self.para_dict['size'], self.para_dict['size'])}]

        self.gaussian_transform = [{'mu':self.para_dict['a_mu'], 'sigma':self.para_dict['a_sigma'],
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                    {'mu':self.para_dict['b_mu'], 'sigma':self.para_dict['b_sigma'],
                                     'size':(self.para_dict['size'], self.para_dict['size'])}]
        self.slight_transform = [{'degrees': self.para_dict['noise_level'], 
                                     'translate': [0.02*self.para_dict['noise_level'], 0.02*self.para_dict['noise_level']],
                                     'scale':[1-0.02*self.para_dict['noise_level'], 1-0.02*self.para_dict['noise_level']], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                    {'degrees': self.para_dict['noise_level'], 
                                     'translate': [0.02*self.para_dict['noise_level'], 0.02*self.para_dict['noise_level']],
                                     'scale':[1-0.02*self.para_dict['noise_level'], 1-0.02*self.para_dict['noise_level']], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])}]
        self.severe_transform = [{'degrees':self.para_dict['severe_rotation'], 
                                     'translate':[self.para_dict['severe_translation'], self.para_dict['severe_translation']],
                                     'scale':[1-self.para_dict['severe_scaling'], 1+self.para_dict['severe_scaling']], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                    {'degrees':self.para_dict['severe_rotation'], 
                                     'translate':[self.para_dict['severe_translation'], self.para_dict['severe_translation']],
                                     'scale':[1-self.para_dict['severe_scaling'], 1+self.para_dict['severe_scaling']], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])}]

        if self.para_dict['noise_type'] == 'normal':
            self.noise_transform = self.normal_transform
        elif self.para_dict['noise_type'] == 'gaussian':
            self.noise_transform = self.gaussian_transform
        elif self.para_dict['noise_type'] == 'slight':
            self.noise_transform = self.slight_transform
        elif self.para_dict['noise_type'] == 'severe': 
            self.noise_transform = self.severe_transform
        else:
            raise NotImplementedError('New Noise Has not been Implemented')

        if self.para_dict['dataset'] == 'brats2021':
            assert self.para_dict['source_domain'] in ['t1', 't2', 'flair']
            assert self.para_dict['target_domain'] in ['t1', 't2', 'flair']
            self.train_dataset = BraTS2021(root=self.para_dict['data_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           noise_type=self.para_dict['noise_type'],
                                           learn_mode='train',
                                           transform_data=self.noise_transform,
                                           client_weights=self.para_dict['clients_data_weight'],
                                           data_mode=self.para_dict['data_mode'],
                                           data_num=self.para_dict['data_num'],
                                           data_paired_weight=self.para_dict['data_paired_weight'],
                                           data_moda_ratio=self.para_dict['data_moda_ratio'],
                                           data_moda_case=self.para_dict['data_moda_case'])
            self.valid_dataset = BraTS2021(root=self.para_dict['valid_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           noise_type='normal',
                                           learn_mode='test',
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           transform_data=self.normal_transform,
                                           data_mode='paired')
            self.assigned_dataset = BraTS2021(root=self.para_dict['valid_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           noise_type='severe',
                                           learn_mode='test',
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           transform_data=self.severe_transform,
                                           data_mode='paired',
                                           assigned_data=self.para_dict['single_img_infer'],
                                           assigned_images=self.para_dict['assigned_images']) 
        elif self.para_dict['dataset'] == 'ixi':
            assert self.para_dict['source_domain'] in ['t2', 'pd']
            assert self.para_dict['target_domain'] in ['t2', 'pd']
            self.train_dataset = IXI(root=self.para_dict['data_path'],
                                    modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                    extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                    noise_type=self.para_dict['noise_type'],
                                    learn_mode='train',
                                    transform_data=self.noise_transform,
                                    data_mode=self.para_dict['data_mode'],
                                    data_num=self.para_dict['data_num'],
                                    data_paired_weight=self.para_dict['data_paired_weight'],
                                    client_weights=self.para_dict['clients_data_weight'],
                                    dataset_splited=True,
                                    data_moda_ratio=self.para_dict['data_moda_ratio'],
                                    data_moda_case=self.para_dict['data_moda_case'])
            self.valid_dataset = IXI(root=self.para_dict['data_path'],
                                    modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                    extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                    noise_type='normal',
                                    learn_mode='test',
                                    transform_data=self.normal_transform,
                                    data_mode='paired',
                                    dataset_splited=True)
            self.assigned_dataset = IXI(root=self.para_dict['data_path'],
                                     modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                     extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                     noise_type='normal',
                                     learn_mode='test',
                                     transform_data=self.severe_transform,
                                     data_mode='paired',
                                     dataset_splited=False,
                                     assigned_data=self.para_dict['single_img_infer'],
                                     assigned_images=self.para_dict['assigned_images']) 
        else:
            raise NotImplementedError('This Dataset Has Not Been Implemented Yet')

        self.client_data_list = self.train_dataset.client_data_indices
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.para_dict['batch_size'],
                                       drop_last=True,
                                       num_workers=self.para_dict['num_workers'],
                                       sampler=SubsetRandomSampler(self.client_data_list[0]))
        self.valid_loader = DataLoader(self.valid_dataset, num_workers=self.para_dict['num_workers'],
                                 batch_size=self.para_dict['batch_size'], shuffle=False)
        self.assigned_loader = DataLoader(self.assigned_dataset, num_workers=self.para_dict['num_workers'],
                                 batch_size=1, shuffle=False)

    def init_model(self):
        if self.para_dict['model'] == 'cyclegan':
            self.trainer = CycleGAN(self.para_dict, self.train_loader, self.valid_loader,
                                    self.assigned_loader, self.device, self.file_path)
        elif self.para_dict['model'] == 'munit':
            self.trainer = Munit(self.para_dict, self.train_loader, self.valid_loader,
                                    self.assigned_loader, self.device, self.file_path)
        elif self.para_dict['model'] == 'unit':
            self.trainer = Unit(self.para_dict, self.train_loader, self.valid_loader,
                                    self.assigned_loader, self.device, self.file_path)
        else:
            raise ValueError('Model is invalid!')

        if self.para_dict['load_model']:
            self.load_models()
            print('load model: {}'.format(self.para_dict['load_model_dir']))

    def load_models(self):
        if self.para_dict['model'] == 'cyclegan':
            gener_from_a_to_b = load_model(self.trainer.generator_from_a_to_b, self.para_dict['load_model_dir'], 'g_from_a_to_b')
            gener_from_b_to_a = load_model(self.trainer.generator_from_b_to_a, self.para_dict['load_model_dir'], 'g_from_b_to_a')
            discr_from_a_to_b = load_model(self.trainer.discriminator_from_a_to_b, self.para_dict['load_model_dir'], 'd_from_a_to_b')
            discr_from_b_to_a = load_model(self.trainer.discriminator_from_b_to_a, self.para_dict['load_model_dir'], 'd_from_b_to_a')
            self.trainer.set_model(gener_from_a_to_b, gener_from_b_to_a, discr_from_a_to_b, discr_from_b_to_a)

        elif self.para_dict['model'] == 'munit' or self.para_dict['model'] == 'unit':
            gener_from_a_to_b_enc = load_model(self.trainer.generator_from_a_to_b_enc, self.para_dict['load_model_dir'], 'g_from_a_to_b_enc')
            gener_from_a_to_b_dec = load_model(self.trainer.generator_from_a_to_b_dec, self.para_dict['load_model_dir'], 'g_from_a_to_b_dec')
            gener_from_b_to_a_enc = load_model(self.trainer.generator_from_b_to_a_enc, self.para_dict['load_model_dir'], 'g_from_b_to_a_enc')
            gener_from_b_to_a_dec = load_model(self.trainer.generator_from_b_to_a_dec, self.para_dict['load_model_dir'], 'g_from_b_to_a_dec')
            discr_from_a_to_b = load_model(self.trainer.discriminator_from_a_to_b, self.para_dict['load_model_dir'], 'd_from_a_to_b')
            discr_from_b_to_a = load_model(self.trainer.discriminator_from_b_to_a, self.para_dict['load_model_dir'], 'd_from_b_to_a')
            self.trainer.set_model(gener_from_a_to_b_enc, gener_from_a_to_b_dec, gener_from_b_to_a_enc, gener_from_b_to_a_dec, discr_from_a_to_b, discr_from_b_to_a)

    def save_models(self):
        if self.para_dict['model'] == 'cyclegan':
            gener_from_a_to_b, gener_from_b_to_a, discr_from_a_to_b, discr_from_b_to_a = self.trainer.get_model()
            save_model(gener_from_a_to_b, '{}/checkpoint/g_from_a_to_b'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(gener_from_b_to_a, '{}/checkpoint/g_from_b_to_a'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(discr_from_a_to_b, '{}/checkpoint/d_from_a_to_b'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(discr_from_b_to_a, '{}/checkpoint/d_from_b_to_a'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))

        elif self.para_dict['model'] == 'munit' or self.para_dict['model'] == 'unit':
            gener_from_a_to_b_enc, gener_from_a_to_b_dec, gener_from_b_to_a_enc, gener_from_b_to_a_dec, discr_from_a_to_b, discr_from_b_to_a = self.trainer.get_model()
            save_model(gener_from_a_to_b_enc, '{}/checkpoint/g_from_a_to_b_enc'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(gener_from_a_to_b_dec, '{}/checkpoint/g_from_a_to_b_dec'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(gener_from_b_to_a_enc, '{}/checkpoint/g_from_b_to_a_enc'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(gener_from_b_to_a_dec, '{}/checkpoint/g_from_b_to_a_dec'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(discr_from_a_to_b, '{}/checkpoint/d_from_a_to_b'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(discr_from_b_to_a, '{}/checkpoint/d_from_b_to_a'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))

    def work_flow(self):
        self.trainer.train_epoch()

        fake_b_value, fake_a_value = self.trainer.evaluation(direction=self.para_dict['direction'])
        psnr_b, psnr_a = 0., 0.

        # evaluation from a to b
        if self.para_dict['direction'] == 'from_a_to_b' or self.para_dict['direction'] == 'both':
            mae_b, psnr_b, ssim_b, fid_b = fake_b_value

            infor = '[Epoch {}/{}] [{} -> {}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
                self.epoch+1, self.para_dict['num_epoch'], self.para_dict['source_domain'], self.para_dict['target_domain'], mae_b, psnr_b, ssim_b)

            if self.para_dict['fid']:
                infor = '{} fid: {:.4f}'.format(infor, fid_b)
            print(infor)
        
            if self.para_dict['save_log']:
                save_log(infor, self.file_path, description='_model_from_a_to_b')

        # evaluation from b to a
        if self.para_dict['direction'] == 'from_b_to_a' or self.para_dict['direction'] == 'both':
            mae_a, psnr_a, ssim_a, fid_a = fake_a_value

            infor = '[Epoch {}/{}] [{} -> {}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
                self.epoch+1, self.para_dict['num_epoch'], self.para_dict['target_domain'], self.para_dict['source_domain'], mae_a, psnr_a, ssim_a)

            if self.para_dict['fid']:
                infor = '{} fid: {:.4f}'.format(infor, fid_a)
            print(infor)

        if self.para_dict['save_log']:
            save_log(infor, self.file_path, description='_model_from_b_to_a')


        if self.para_dict['plot_distribution']:
            save_img_path = '{}/sample_distribution'.format(self.file_path)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            save_img_path = '{}/epoch_{}.png'.format(save_img_path, self.epoch+1)
            self.trainer.visualize_feature(self.epoch+1, save_img_path, self.train_loader)

        if self.para_dict['save_img']:
            save_img_path = '{}/images/epoch_{}'.format(self.file_path, self.epoch+1)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            self.trainer.infer_images(save_img_path, self.valid_loader)

        if self.para_dict['single_img_infer']:
            save_img_path = '{}/images_assigned/epoch_{}'.format(self.file_path, self.epoch+1)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            self.trainer.infer_images(save_img_path, self.assigned_loader)

        if self.para_dict['save_model']:
            if psnr_b > self.best_psnr_b and psnr_a > self.best_psnr_a:
                self.save_models()
                self.best_psnr_b = psnr_b
                self.best_psnr_a = psnr_a

    def run_work_flow(self):
        self.load_config()
        self.preliminary()
        self.load_data()
        self.init_model()
        print('---------------------')

        for epoch in range(self.para_dict['num_epoch']):
            self.epoch = epoch
            self.work_flow()
            
        print('work dir: {}'.format(self.file_path))
        with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
            print('\n---> work dir {}'.format(self.file_path), file=f)
            print(self.args, file=f)
        print('---------------------')

