import torch
import yaml
import os
from tools.utilize import *
from data_io.brats import BraTS2021
from data_io.ixi import IXI
from torch.utils.data import DataLoader
from architecture.centralized.train import CentralizedTrain
from architecture.nirps.cyclegan import NIRPSCycleGAN

import warnings
warnings.filterwarnings("ignore")


class NIRPS(CentralizedTrain):
    def __init__(self, args):
        super(NIRPS, self).__init__(args=args)
        self.args = args

    def load_config(self):
        # load nirps
        with open('./configuration/nirps/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_kaid = yaml.load(f, Loader=yaml.SafeLoader)
        # load basic 
        with open('./configuration/architecture/3_dataset_base/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_dataset = yaml.load(f, Loader=yaml.SafeLoader)
        with open('configuration/architecture/2_train_base/centralized_training.yaml', 'r') as f:
            config_train = yaml.load(f, Loader=yaml.SafeLoader)
        with open('configuration/architecture/1_model_base/{}.yaml'.format(self.args.model), 'r') as f:
            config_model = yaml.load(f, Loader=yaml.SafeLoader)

        config = override_config(config_model, config_train)
        config = override_config(config, config_dataset)
        config = override_config(config, config_kaid)
        self.para_dict = merge_config(config, self.args)
        self.args = extract_config(self.args)

    def preliminary(self):
        return super().preliminary()

    def setup_folder(self):
        # fp: file path
        self.dataset_fp = os.path.join(self.para_dict['generated_dataset_dir'], self.para_dict['dataset'])
        create_folders(self.dataset_fp)

        self.source_modality_fp = os.path.join(self.dataset_fp, self.para_dict['source_domain'])
        create_folders(self.source_modality_fp)

        self.target_modality_fp = os.path.join(self.dataset_fp, self.para_dict['target_domain'])
        create_folders(self.target_modality_fp)

        self.model_source_fp = os.path.join(self.source_modality_fp, self.para_dict['model']) 
        self.model_target_fp = os.path.join(self.target_modality_fp, self.para_dict['model'])
        create_folders(self.model_source_fp)
        create_folders(self.model_target_fp)

        self.model_source_gt_fp = os.path.join(self.model_source_fp, 'gt')
        self.model_target_gt_fp = os.path.join(self.model_target_fp, 'gt')

        for i in range(self.para_dict['num_epoch']):
            epoch_model_source_fp = os.path.join(self.model_source_fp, str(i)) 
            epoch_model_target_fp = os.path.join(self.model_target_fp, str(i)) 
            create_folders(epoch_model_source_fp)
            create_folders(epoch_model_target_fp)
            
    def load_data(self):
        self.normal_transform = [{'degrees':0, 'translate':[0.00, 0.00],
                                     'scale':[1.00, 1.00], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                 {'degrees':0, 'translate':[0.00, 0.00],
                                  'scale':[1.00, 1.00], 
                                  'size':(self.para_dict['size'], self.para_dict['size'])}]

        if self.para_dict['dataset'] == 'brats2021':
            assert self.para_dict['source_domain'] in ['t1', 't2', 'flair']
            assert self.para_dict['target_domain'] in ['t1', 't2', 'flair']

            self.train_dataset = BraTS2021(root=self.para_dict['data_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           noise_type='normal',
                                           learn_mode='train',
                                           transform_data=self.normal_transform,
                                           client_weights=[1.0],
                                           data_mode=self.para_dict['data_mode'],
                                           data_num=self.para_dict['data_num'],
                                           data_paired_weight=1.0,
                                           data_moda_ratio=1.0,
                                           data_moda_case='case1')

            self.valid_dataset = BraTS2021(root=self.para_dict['valid_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           noise_type='normal',
                                           learn_mode='test',
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           transform_data=self.normal_transform,
                                           data_mode='paired',
                                           assigned_data=True,
                                           assigned_images=None) 
            
        elif self.para_dict['dataset'] == 'ixi':
            assert self.para_dict['source_domain'] in ['t2', 'pd']
            assert self.para_dict['target_domain'] in ['t2', 'pd']

            self.train_dataset = IXI(root=self.para_dict['data_path'],
                                     modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                     extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                     noise_type='normal',
                                     learn_mode='train',
                                     transform_data=self.normal_transform,
                                     data_mode=self.para_dict['data_mode'],
                                     data_num=self.para_dict['data_num'],
                                     data_paired_weight=1.0,
                                     client_weights=[1.0],
                                     dataset_splited=True,
                                     data_moda_ratio=1.0,
                                     data_moda_case='case1')

            self.valid_dataset = IXI(root=self.para_dict['data_path'],
                                     modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                     extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                     noise_type='normal',
                                     learn_mode='test',
                                     transform_data=self.normal_transform,
                                     data_mode='paired',
                                     dataset_splited=True) 
        else:
            raise NotImplementedError('This Dataset Has Not Been Implemented Yet')

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.para_dict['batch_size'],
                                       num_workers=self.para_dict['num_workers'],
                                       shuffle=True)

        self.valid_loader = DataLoader(self.valid_dataset, 
                                       num_workers=self.para_dict['num_workers'],
                                       batch_size=1, 
                                       shuffle=False)
        
        self.assigned_loader = None

    def init_model(self):
        if self.para_dict['model'] == 'cyclegan':
            self.trainer = NIRPSCycleGAN(self.para_dict, self.train_loader, self.valid_loader,
                                    self.assigned_loader, self.device, self.file_path)
        else:
            raise ValueError('Model is invalid!')

        if self.para_dict['load_model']:
            self.load_models()
            print('load model: {}'.format(self.para_dict['load_model_dir']))

    def save_models(self, fp=None, epoch=None):
        if self.para_dict['model'] == 'cyclegan':
            gener_from_a_to_b, gener_from_b_to_a, discr_from_a_to_b, discr_from_b_to_a = self.trainer.get_model()
            save_model(gener_from_a_to_b, '{}/checkpoint/g_from_a_to_b'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(gener_from_b_to_a, '{}/checkpoint/g_from_b_to_a'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
        else:
            raise ValueError('Model is invalid!')

    def work_flow(self):
        self.trainer.train_epoch()
        # evaluation from a to b
        if self.para_dict['general_evaluation']:
            (source_mae, source_psnr, source_ssim, source_fid, target_mae, target_psnr, target_ssim, target_fid) = self.trainer.evaluation(direction='both')

            src_infor = '[Epoch {}/{}] [{}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f} fid: {:.4f}'.format(
                self.epoch+1, self.para_dict['num_epoch'], self.para_dict['source_domain'], source_mae, source_psnr, source_ssim, source_fid)

            tag_infor = '[Epoch {}/{}] [{}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f} fid: {:.4f}'.format(
                self.epoch+1, self.para_dict['num_epoch'], self.para_dict['target_domain'], target_mae, target_psnr, target_ssim, target_fid)
        
            print(src_infor)
            print(tag_infor)
        
            save_log(src_infor, self.file_path, description='both')
            save_log(tag_infor, self.file_path, description='both')
        
        for i in range(int(self.para_dict['num_epoch'])):
            epoch_model_source_fp = os.path.join(self.model_source_fp, str(i)) 
            epoch_model_target_fp = os.path.join(self.model_target_fp, str(i)) 
            self.save_models(fp=epoch_model_source_fp, epoch=i)
            self.save_models(fp=epoch_model_target_fp, epoch=i)
            self.trainer.infer_nirps_generated(src_epoch_path=epoch_model_source_fp,
                                               tag_epoch_path=epoch_model_target_fp,
                                               data_loader=self.valid_loader)
        
        self.trainer.infer_nirps_gt(src_gt_path=self.model_source_gt_fp,
                                    tag_gt_path=self.model_target_gt_fp,
                                    data_loader=self.valid_loader)
            
    def run_work_flow(self):
        self.load_config()
        self.preliminary()
        self.setup_folder()
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