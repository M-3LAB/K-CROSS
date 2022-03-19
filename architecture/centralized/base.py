import torch
import numpy as np

from tools.utilize import *
from metrics.metrics import mae, psnr, ssim, fid
from tools.visualize import plot_sample

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


__all__ = ['BASE']

class BASE():
    def __init__(self, config, train_loader, valid_loader, assigned_loader,
                 device, file_path, batch_limit_weight=1.0):

        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.assigned_loader = assigned_loader
        self.device = device
        self.file_path = file_path
        self.batch_limit_weight = batch_limit_weight
        self.angle_list = config['angle_list']
        self.translation_list = config['translation_list']
        self.scaling_list = config['scaling_list']
        self.batch_size = config['batch_size']
        self.valid = 1
        self.fake = 0

        # fid stats
        self.fid_stats_from_a_to_b = '{}/{}/{}.npz'.format(
            self.config['fid_dir'], self.config['dataset'], self.config['target_domain'])
        self.fid_stats_from_b_to_a = '{}/{}/{}.npz'.format(
            self.config['fid_dir'], self.config['dataset'], self.config['source_domain'])

        # differential privacy
        if self.config['diff_privacy']:
            self.clip_bound = self.config['clip_bound']
            self.sensitivity = self.config['sensitivity']
            self.noise_multiplier = self.config['noise_multiplier']

        # model, two modality, 1 and 2, the aim is to generate 2 from 1
        self.generator_from_a_to_b_enc = None
        self.generator_from_b_to_a_enc = None
        self.generator_from_a_to_b_dec = None
        self.generator_from_b_to_a_dec = None
        self.discriminator_from_a_to_b = None
        self.discriminator_from_b_to_a = None

        # loss
        self.criterion_recon = torch.nn.L1Loss().to(self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)
        self.criterion_gan_from_a_to_b = torch.nn.MSELoss().to(device)
        self.criterion_gan_from_b_to_a = torch.nn.MSELoss().to(device)
        self.criterion_pixelwise_from_a_to_b = torch.nn.L1Loss().to(device)
        self.criterion_pixelwise_from_b_to_a = torch.nn.L1Loss().to(device)
        self.criterion_identity = torch.nn.L1Loss().to(device)
        self.criterion_sr = torch.nn.L1Loss().to(device)
        self.criterion_rotation = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_rotation = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_translation = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_scaling = torch.nn.BCEWithLogitsLoss().to(device)

        # differential privacy
        if self.config['diff_privacy']:
            pass

        # optimizer
        self.optimizer_generator = None
        self.optimizer_discriminator_from_a_to_b = None
        self.optimizer_discriminator_from_b_to_a = None

        self.lr_scheduler_generator = None
        self.lr_scheduler_discriminator_from_a_to_b = None
        self.lr_scheduler_discriminator_from_b_to_a = None

        self.batch_limit = int(self.config['data_num'] * self.batch_limit_weight / self.config['batch_size'])
        if self.config['debug']:
            self.batch_limit = 2

    def calculate_basic_gan_loss(self, images):
        pass

    def train_epoch(self, inf=''):
        for i, batch in enumerate(self.train_loader):
            if i > self.batch_limit:
                break

            """
            Train Generators
            """
            # differential privacy, train with real
            if self.config['diff_privacy']:
                self.dynamic_hook_function = self.dummy_hook

            self.optimizer_generator.zero_grad()

            imgs, tmps = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs

            # gan loss
            loss_gan_basic = self.calculate_basic_gan_loss([imgs, tmps])

            # total loss
            loss_generator_total = loss_gan_basic

            # idn loss
            if self.config['identity']:
                loss_identity_fake_b = self.criterion_identity(fake_b, fake_fake_a)
                loss_identity_fake_a = self.criterion_identity(fake_a, fake_fake_b)
                loss_identity = self.config['lambda_identity'] * (loss_identity_fake_a + loss_identity_fake_b)
                loss_generator_total += loss_identity

            loss_generator_total.backward()

            # differential privacy
            if self.config['diff_privacy']:
                # sanitize the gradients passed to the generator
                self.dynamic_hook_function = self.diff_privacy_conv_hook

            self.optimizer_generator.step()

            """
            Train Discriminator
            """
            self.optimizer_discriminator_from_a_to_b.zero_grad()
            self.optimizer_discriminator_from_b_to_a.zero_grad()

            loss_discriminator_from_a_to_b = self.discriminator_from_a_to_b.compute_loss(
                real_a, self.valid) + self.discriminator_from_a_to_b.compute_loss(fake_a.detach(), self.fake)
            loss_discriminator_from_b_to_a = self.discriminator_from_b_to_a.compute_loss(
                real_b, self.valid) + self.discriminator_from_b_to_a.compute_loss(fake_b.detach(), self.fake)

            loss_discriminator_from_a_to_b_total = loss_discriminator_from_a_to_b
            loss_discriminator_from_b_to_a_total = loss_discriminator_from_b_to_a

            loss_discriminator_from_a_to_b_total.backward(retain_graph=True)
            loss_discriminator_from_b_to_a_total.backward(retain_graph=True)

            self.optimizer_discriminator_from_b_to_a.step()
            self.optimizer_discriminator_from_a_to_b.step()

            # print log
            infor = '\r{}[Batch {}/{}] [Gen loss: {:.4f}] [Dis loss: {:.4f}, {:.4f}]'.format(inf, i, self.batch_limit,
                        loss_generator_total.item(), loss_discriminator_from_a_to_b.item(), loss_discriminator_from_b_to_a.item())

            if self.config['identity']:
                infor = '{} [Idn Loss: {:.4f}]'.format(infor, loss_identity.item())

            print(infor, flush=True, end=' ')

        # update learning rates
        self.lr_scheduler_generator.step()
        self.lr_scheduler_discriminator_from_a_to_b.step()
        self.lr_scheduler_discriminator_from_b_to_a.step()


    def collect_compute_result_for_evaluation(self):
        # initialize fake_b_list
        fake_b_list = torch.randn(self.config['batch_size'], 1, self.config['size'], self.config['size'])
        fake_a_list = torch.randn(self.config['batch_size'], 1, self.config['size'], self.config['size'])
        # to reduce gpu memory for evaluation
        mae_list, psnr_list, ssim_list = [], [], []
        for i, batch in enumerate(self.valid_loader):
            imgs, tmps = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs

            if self.config['fid']:
                fake_b_list = concate_tensor_lists(fake_b_list, fake_b, i)
                fake_a_list = concate_tensor_lists(fake_a_list, fake_a, i)

            mae_value = mae(real_b, fake_b) 
            psnr_value = psnr(real_b, fake_b)
            ssim_value = ssim(real_b, fake_b)
            mae_list.append(mae_value)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)

        
        return fake_b_list, mae_list, psnr_list,  ssim_list

    @torch.no_grad()
    def evaluation(self, direction='from_a_to_b'):
        
        fake_a_list = torch.randn(self.config['batch_size'], 1, self.config['size'], self.config['size'])
        fake_b_list = torch.randn(self.config['batch_size'], 1, self.config['size'], self.config['size'])
        mae_a_list, psnr_a_list, ssim_a_list = [], [], []
        mae_b_list, psnr_b_list, ssim_b_list = [], [], []
        mae_a_value, psnr_a_value, ssim_a_value, fid_a_value = 0., 0., 0., 0.
        mae_b_value, psnr_b_value, ssim_b_value, fid_b_value = 0., 0., 0., 0.

        for i, batch in enumerate(self.valid_loader):
            imgs, tmps = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs

            if direction == 'from_a_to_b' or direction == 'both':
                mae_value = mae(real_b, fake_b) 
                psnr_value = psnr(real_b, fake_b)
                ssim_value = ssim(real_b, fake_b)
                mae_b_list.append(mae_value)
                psnr_b_list.append(psnr_value)
                ssim_b_list.append(ssim_value)

                if self.config['fid']:
                    fake_b_list = concate_tensor_lists(fake_b_list, fake_b, i)

            if direction == 'from_b_to_a' or direction == 'both':
                mae_value = mae(real_a, fake_a) 
                psnr_value = psnr(real_a, fake_a)
                ssim_value = ssim(real_a, fake_a)
                mae_a_list.append(mae_value)
                psnr_a_list.append(psnr_value)
                ssim_a_list.append(ssim_value)

                if self.config['fid']:
                    fake_a_list = concate_tensor_lists(fake_a_list, fake_a, i)

        if direction == 'from_a_to_b' or direction == 'both':
            mae_b_value = average(mae_b_list)
            psnr_b_value = average(psnr_b_list)
            ssim_b_value = average(ssim_b_list)

            if self.config['fid']:
                fid_b_value = fid(fake_b_list, self.config['batch_size_inceptionV3'],
                                self.config['target_domain'], self.fid_stats_from_a_to_b, self.device)

        if direction == 'from_b_to_a' or direction == 'both':
            mae_a_value = average(mae_a_list)
            psnr_a_value = average(psnr_a_list)
            ssim_a_value = average(ssim_a_list)

            if self.config['fid']:
                fid_a_value = fid(fake_a_list, self.config['batch_size_inceptionV3'],
                                self.config['source_domain'], self.fid_stats_from_b_to_a, self.device)

        return [[mae_b_value, psnr_b_value, ssim_b_value, fid_b_value],
                [mae_a_value, psnr_a_value, ssim_a_value, fid_a_value]]

    def get_model(self, description='centralized'):
        return self.generator_from_a_to_b_enc, self.generator_from_b_to_a_enc, self.generator_from_a_to_b_dec,\
             self.generator_from_b_to_a_dec, self.discriminator_from_a_to_b, self.discriminator_from_b_to_a

    def set_model(self, gener_from_a_to_b_enc, gener_from_a_to_b_dec, gener_from_b_to_a_enc,\
                gener_from_b_to_a_dec, discr_from_a_to_b, discr_from_b_to_a):
        self.generator_from_a_to_b_enc = gener_from_a_to_b_enc
        self.generator_from_a_to_b_dec = gener_from_a_to_b_dec
        self.generator_from_b_to_a_enc = gener_from_b_to_a_enc
        self.generator_from_b_to_a_dec = gener_from_b_to_a_dec
        self.discriminator_from_a_to_b = discr_from_a_to_b
        self.discriminator_from_b_to_a = discr_from_b_to_a

    def collect_generated_images(self, batch):
        pass


    def collect_feature(self, batch):
        pass

    @torch.no_grad()
    def infer_images(self, save_img_path, data_loader):
        for i, batch in enumerate(data_loader):
            imgs, tmps = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs
            if i <= self.config['num_img_save']:
                img_path = '{}/{}-slice-{}'.format(
                    save_img_path, batch['name_a'][0], batch['slice_num'].numpy()[0])

                mae_value = mae(real_b, fake_b).item() 
                psnr_value = psnr(real_b, fake_b).item()
                ssim_value = ssim(real_b, fake_b).item()
                    
                img_all = torch.cat((real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b), 0)
                save_image(img_all, 'all_m_{:.4f}_p_{:.4f}_s_{:.4f}.png'.format(mae_value, psnr_value, ssim_value), img_path)

                save_image(real_a, 'real_a.png', img_path)
                save_image(real_b, 'real_b.png', img_path)
                save_image(fake_a, 'fake_a.png', img_path)
                save_image(fake_b, 'fake_b.png', img_path)
                save_image(fake_fake_a, 'fake_fake_a.png', img_path)
                save_image(fake_fake_b, 'fake_fake_b.png', img_path)

    @torch.no_grad()
    def visualize_feature(self, epoch, save_img_path, data_loader):
        real_a, fake_a, real_b, fake_b = [], [], [], []
        for i, batch in enumerate(data_loader):
            if i == int(self.config['plot_num_sample']):
                break
            real_a_feature, fake_a_feature, real_b_feature, fake_b_feature = self.collect_feature(batch=batch)

            real_a_feature = np.mean(real_a_feature.cpu().detach().numpy().reshape(
                len(batch[self.config['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            fake_a_feature = np.mean(fake_a_feature.cpu().detach().numpy().reshape(
                len(batch[self.config['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            real_b_feature = np.mean(real_b_feature.cpu().detach().numpy().reshape(
                len(batch[self.config['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            fake_b_feature = np.mean(fake_b_feature.cpu().detach().numpy().reshape(
                len(batch[self.config['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))

            if i == 0:
                real_a = real_a_feature
                fake_a = fake_a_feature
                real_b = real_b_feature
                fake_b = fake_b_feature
            else:
                real_a = np.concatenate([real_a, real_a_feature], axis=0)
                fake_a = np.concatenate([fake_a, fake_a_feature], axis=0)
                real_b = np.concatenate([real_b, real_b_feature], axis=0)
                fake_b = np.concatenate([fake_b, fake_b_feature], axis=0)

        plot_sample(real_a, fake_a, real_b, fake_b, step=epoch, img_path=save_img_path, descript='Epoch')

        with open(save_img_path.replace('.png', '.npy'), 'wb') as f:
            np.save(f, np.array([real_a, fake_a, real_b, fake_b]))


    def master_hook_adder(self, module, grad_input, grad_output):
        # global dynamic_hook_function
        return self.dynamic_hook_function(module, grad_input, grad_output)

    def dummy_hook(self, module, grad_input, grad_output):
        pass

    # differential privacy, train with real
    def modify_gradnorm_conv_hook(self, module, grad_input, grad_output):
        # get grad wrt. input (image)
        grad_wrt_image = grad_input[0]
        grad_input_shape = grad_wrt_image.size()
        batchsize = grad_input_shape[0]
        clip_bound_ = self.clip_bound / batchsize  # account for the 'sum' operation in GP

        grad_wrt_image = grad_wrt_image.view(batchsize, -1)
        grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

        # clip
        clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
        clip_coef = clip_coef.unsqueeze(-1)
        grad_wrt_image = clip_coef * grad_wrt_image
        grad_input_new = [grad_wrt_image.view(grad_input_shape)]
        for i in range(len(grad_input)-1):
            grad_input_new.append(grad_input[i+1])

        return tuple(grad_input_new)

    # differential privacy, train with real
    def diff_privacy_conv_hook(self, module, grad_input, grad_output):
        # global noise_multiplier
        # get grad wrt. input (image)
        grad_wrt_image = grad_input[0]
        grad_input_shape = grad_wrt_image.size()
        batchsize = grad_input_shape[0]
        clip_bound_ = self.clip_bound / batchsize

        grad_wrt_image = grad_wrt_image.view(batchsize, -1)
        grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

        # clip
        clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        clip_coef = clip_coef.unsqueeze(-1)
        grad_wrt_image = clip_coef * grad_wrt_image

        # add noise
        noise = clip_bound_ * self.noise_multiplier * self.sensitivity * torch.randn_like(grad_wrt_image)
        grad_wrt_image = grad_wrt_image + noise
        grad_input_new = [grad_wrt_image.view(grad_input_shape)]
        for i in range(len(grad_input)-1):
            grad_input_new.append(grad_input[i+1])

        return tuple(grad_input_new)

