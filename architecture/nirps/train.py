from operator import gt
import yaml
import os
import shutil
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
                                           data_mode='unpaired',
                                           data_num=self.para_dict['data_num'])
            self.valid_dataset = BraTS2021(root=self.para_dict['valid_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           noise_type='normal',
                                           learn_mode='test',
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           transform_data=self.normal_transform,
                                           data_mode='paired')
        elif self.para_dict['dataset'] == 'ixi':
            assert self.para_dict['source_domain'] in ['t2', 'pd']
            assert self.para_dict['target_domain'] in ['t2', 'pd']

            self.train_dataset = IXI(root=self.para_dict['data_path'],
                                     modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                     extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                     noise_type='normal',
                                     learn_mode='train',
                                     transform_data=self.normal_transform,
                                     data_mode='unpaired',
                                     data_num=self.para_dict['data_num'],
                                     dataset_splited=True)
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
                                       batch_size=1, 
                                       num_workers=self.para_dict['num_workers'],
                                       shuffle=False)

    def init_model(self):
        if self.para_dict['model'] == 'cyclegan':
            self.trainer = NIRPSCycleGAN(self.para_dict, self.train_loader, self.valid_loader,
                                    self.assigned_loader, self.device, self.file_path)
        else:
            raise ValueError('Model is Invalid!')

        if self.para_dict['load_model']:
            self.load_models()
            print('load model: {}'.format(self.para_dict['load_model_dir']))

    def save_models(self):
        if self.para_dict['model'] == 'cyclegan':
            gener_from_a_to_b, gener_from_b_to_a, discr_from_a_to_b, discr_from_b_to_a = self.trainer.get_model()
            save_model(gener_from_a_to_b, '{}/checkpoint/g_from_a_to_b'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
            save_model(gener_from_b_to_a, '{}/checkpoint/g_from_b_to_a'.format(self.file_path), 'epoch_{}'.format(self.epoch+1))
        else:
            raise ValueError('Model is Invalid!')

    def work_flow(self):
        # train model
        self.trainer.train_epoch()
        # evaluation
        if self.para_dict['general_evaluation']:
            [[mae_b, psnr_b, ssim_b, fid_b], [mae_a, psnr_a, ssim_a, fid_a]]= self.trainer.evaluation(direction='both')

            infor_a = '[Epoch {}/{}] [{}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f} fid: {:.4f}'.format(
                self.epoch+1, self.para_dict['num_epoch'], self.para_dict['source_domain'], mae_a, psnr_a, ssim_a, fid_a)
            infor_b = '[Epoch {}/{}] [{}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f} fid: {:.4f}'.format(
                self.epoch+1, self.para_dict['num_epoch'], self.para_dict['target_domain'], mae_b, psnr_b, ssim_b, fid_b)
        
            print(infor_a)
            print(infor_b)
        
            save_log(infor_a, self.file_path, description='both')
            save_log(infor_b, self.file_path, description='both')
        
        # generate nirps dataset
        gt_a_path = '{}/nirps_dataset/{}/{}/gt'.format(self.file_path, self.para_dict['dataset'], self.para_dict['source_domain'])
        gt_b_path = '{}/nirps_dataset/{}/{}/gt'.format(self.file_path, self.para_dict['dataset'], self.para_dict['target_domain'])
        save_a_path = '{}/nirps_dataset/{}/{}/{}/epoch_{}'.format(self.file_path, self.para_dict['dataset'], self.para_dict['source_domain'], self.para_dict['model'], self.epoch+1)
        save_b_path = '{}/nirps_dataset/{}/{}/{}/epoch_{}'.format(self.file_path, self.para_dict['dataset'], self.para_dict['target_domain'], self.para_dict['model'], self.epoch+1)

        self.trainer.infer_nirps_dataset(save_a_path=save_a_path, save_b_path=save_b_path, 
                                            gt_a_path=gt_a_path, gt_b_path=gt_b_path,
                                            data_loader=self.valid_loader)
        # save model
        # self.save_models()

    def copy_img_into_nirps_dataset(self, data_moda):
        source_path = '{}/nirps_dataset/{}/{}'.format(self.file_path, self.para_dict['dataset'], data_moda)
        target_path = '{}/{}'.format(self.para_dict['nirps_dataset'], self.para_dict['dataset'], data_moda)

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if os.path.exists(source_path):
            shutil.rmtree(target_path)

        shutil.copytree(source_path, target_path)
        print('copy dataset finished, nirps dataset dir: '.format(target_path))


    def run_work_flow(self):
        self.load_config()
        self.preliminary()
        self.load_data()
        self.init_model()
        print('---------------------')

        for epoch in range(self.para_dict['num_epoch']):
            self.epoch = epoch
            self.work_flow()

        # copy generated images to target nirps dataset
        self.copy_img_into_nirps_dataset(self.para_dict['source_domain']) 
        self.copy_img_into_nirps_dataset(self.para_dict['target_domain']) 

        print('work dir: {}'.format(self.file_path))
        with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
            print('\n---> work dir {}'.format(self.file_path), file=f)
            print(self.args, file=f)
        print('---------------------')