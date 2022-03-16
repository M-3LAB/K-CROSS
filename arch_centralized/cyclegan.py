import torch
import numpy as np

from arch_centralized.base import Base
from model.cyclegan.cyclegan import CycleGen, CycleDis
from tools.utilize import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



__all__ = ['CycleGAN']

class CycleGAN(Base):
    def __init__(self, config, train_loader, valid_loader, assigned_loader, 
                 device, file_path, batch_limit_weight=1.0):
        super(CycleGAN, self).__init__(config=config, train_loader=train_loader, valid_loader=valid_loader, assigned_loader=assigned_loader, 
                 device=device, file_path=file_path, batch_limit_weight=batch_limit_weight)

        self.config = config

        # model
        self.generator_from_a_to_b = CycleGen().to(self.device)
        self.generator_from_b_to_a = CycleGen().to(self.device)

        self.discriminator_from_a_to_b = CycleDis(auxiliary_rotation=self.config['auxiliary_rotation'],
                                                    auxiliary_translation=self.config['auxiliary_translation'],
                                                    auxiliary_scaling=self.config['auxiliary_scaling'],
                                                    num_augmentation=self.config['num_augmentation']).to(self.device)

        self.discriminator_from_b_to_a = CycleDis(auxiliary_rotation=self.config['auxiliary_rotation'],
                                                      auxiliary_translation=self.config['auxiliary_translation'],
                                                      auxiliary_scaling=self.config['auxiliary_scaling'],
                                                      num_augmentation=self.config['num_augmentation']).to(self.device)

        # differential privacy
        if self.config['diff_privacy']:
            self.discriminator_from_a_to_b.model[0].register_backward_hook(self.master_hook_adder)
            self.discriminator_from_b_to_a.model[0].register_backward_hook(self.master_hook_adder)

        # optimizer
        self.optimizer_generator_from_a_to_b = torch.optim.Adam(self.generator_from_a_to_b.parameters(),
                                                                lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        self.optimizer_generator_from_b_to_a = torch.optim.Adam(self.generator_from_b_to_a.parameters(),
                                                                lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        self.optimizer_discriminator_from_a_to_b = torch.optim.Adam(self.discriminator_from_a_to_b.parameters(),
                                                                    lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        self.optimizer_discriminator_from_b_to_a = torch.optim.Adam(self.discriminator_from_b_to_a.parameters(),
                                                               lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))


    def train_epoch(self, inf=''):
        Tensor = torch.cuda.FloatTensor

        for i, batch in enumerate(self.train_loader):
            if i > self.batch_limit:
                break
            real_a = batch[self.config['source_domain']].to(self.device)
            real_b = batch[self.config['target_domain']].to(self.device)
            
            # create labels
            valid_a = Tensor(np.ones((real_a.size(0), 1, 1, 1))).to(self.device)
            valid_b = Tensor(np.ones((real_b.size(0), 1, 1, 1))).to(self.device)
            imitation_a = Tensor(np.zeros((real_a.size(0), 1, 1, 1))).to(self.device)
            imitation_b = Tensor(np.zeros((real_b.size(0), 1, 1, 1))).to(self.device)

            """
            Train Generators
            """
            # differential privacy, train with real
            if self.config['diff_privacy']:
                self.dynamic_hook_function = self.dummy_hook

            self.optimizer_generator_from_a_to_b.zero_grad()
            self.optimizer_generator_from_b_to_a.zero_grad()
            # gan loss
            fake_b = self.generator_from_a_to_b(real_a)
            fake_a = self.generator_from_b_to_a(real_b)

            pred_fake_b = self.discriminator_from_a_to_b(x=fake_b)
            pred_fake_a = self.discriminator_from_b_to_a(x=fake_a)

            loss_gan_from_a_to_b = self.criterion_gan_from_a_to_b(pred_fake_b, valid_b)
            loss_gan_from_b_to_a = self.criterion_gan_from_b_to_a(pred_fake_a, valid_a)

            # l1 loss
            fake_fake_a = self.generator_from_b_to_a(fake_b)
            fake_fake_b = self.generator_from_a_to_b(fake_a)
            loss_pixel_from_a_to_b = self.criterion_pixelwise_from_a_to_b(fake_fake_a, real_a)
            loss_pixel_from_b_to_a = self.criterion_pixelwise_from_b_to_a(fake_fake_b, real_b)

            # gan loss
            loss_generator_from_a_to_b = (self.config['lambda_gan'] * loss_gan_from_a_to_b +
                                          self.config['lambda_cyc'] * loss_pixel_from_a_to_b)
            loss_generator_from_b_to_a = (self.config['lambda_gan'] * loss_gan_from_b_to_a +
                                          self.config['lambda_cyc'] * loss_pixel_from_b_to_a)

            loss_generator_total = loss_generator_from_a_to_b + loss_generator_from_b_to_a

            if self.config['identity']:
                loss_identity_fake_b = self.criterion_identity(fake_b, self.generator_from_a_to_b(fake_b))
                loss_identity_fake_a = self.criterion_identity(fake_a, self.generator_from_b_to_a(fake_a))
                loss_identity = self.config['lambda_identity'] * (loss_identity_fake_a + loss_identity_fake_b)
                loss_generator_total += loss_identity

            loss_generator_total.backward()

            # differential privacy
            if self.config['diff_privacy']:
                # sanitize the gradients passed to the generator
                self.dynamic_hook_function = self.diff_privacy_conv_hook

            self.optimizer_generator_from_a_to_b.step()
            self.optimizer_generator_from_b_to_a.step()

            """
            Train Discriminator
            """
            self.optimizer_discriminator_from_a_to_b.zero_grad()
            self.optimizer_discriminator_from_b_to_a.zero_grad()

            # real loss
            pred_real_b = self.discriminator_from_a_to_b(x=real_b)
            pred_real_a = self.discriminator_from_b_to_a(x=real_a)

            loss_real_b = self.criterion_gan_from_a_to_b(pred_real_b, valid_b)
            loss_real_a = self.criterion_gan_from_b_to_a(pred_real_a, valid_a)

            # fake loss
            pred_fake_b = self.discriminator_from_a_to_b(x=fake_b.detach())
            pred_fake_a = self.discriminator_from_b_to_a(x=fake_a.detach())
                
            loss_fake_a = self.criterion_gan_from_b_to_a(pred_fake_a, imitation_a)
            loss_fake_b = self.criterion_gan_from_a_to_b(pred_fake_b, imitation_b)

            # discriminator loss
            loss_discriminator_from_a_to_b = 0.5 * (loss_real_b + loss_fake_b)
            loss_discriminator_from_b_to_a = 0.5 * (loss_real_a + loss_fake_a)
            loss_discriminator_total = loss_discriminator_from_a_to_b + loss_discriminator_from_b_to_a
            
            #loss_discriminator_total.backward(retain_graph=True)
            loss_discriminator_total.backward()
            self.optimizer_discriminator_from_a_to_b.step()
            self.optimizer_discriminator_from_b_to_a.step()

            # print log
            infor = '\r{}[Batch {}/{}] [Gen loss: {:.4f}, {:.4f}] [Dis loss: {:.4f}, {:.4f}]'.format(inf, i, self.batch_limit,
                        loss_generator_from_a_to_b.item(), loss_generator_from_b_to_a.item(),
                        loss_discriminator_from_a_to_b.item(), loss_discriminator_from_b_to_a.item())

            if self.config['identity']:
                infor = '{} [Idn Loss: {:.4f}]'.format(infor, loss_identity.item())

            print(infor, flush=True, end=' ')
            
    def collect_generated_images(self, batch):
        real_a = batch[self.config['source_domain']].to(self.device)
        real_b = batch[self.config['target_domain']].to(self.device)
        
        fake_a = self.generator_from_b_to_a(real_b)
        fake_b = self.generator_from_a_to_b(real_a)        
        fake_fake_a  = self.generator_from_b_to_a(fake_b)
        fake_fake_b = self.generator_from_a_to_b(fake_a)

        imgs = [real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b]
        return imgs, None

    def get_model(self, description='centralized'):
        return self.generator_from_a_to_b, self.generator_from_b_to_a, self.discriminator_from_a_to_b, self.discriminator_from_b_to_a

    def set_model(self, gener_from_a_to_b, gener_from_b_to_a, discr_from_a_to_b, discr_from_b_to_a):
        self.generator_from_a_to_b = gener_from_a_to_b
        self.generator_from_b_to_a = gener_from_b_to_a
        self.discriminator_from_a_to_b = discr_from_a_to_b
        self.discriminator_from_b_to_a = discr_from_b_to_a

    def collect_feature(self, batch):
        real_a = batch[self.config['source_domain']].to(self.device)
        real_b = batch[self.config['target_domain']].to(self.device)

        fake_a = self.generator_from_b_to_a(real_b)
        fake_b = self.generator_from_a_to_b(real_a)

        real_a_feature = self.generator_from_a_to_b.extract_feature(real_a)
        fake_a_feature = self.generator_from_a_to_b.extract_feature(fake_a)
        real_b_feature = self.generator_from_b_to_a.extract_feature(real_b)
        fake_b_feature = self.generator_from_b_to_a.extract_feature(fake_b)

        return real_a_feature, fake_a_feature, real_b_feature, fake_b_feature