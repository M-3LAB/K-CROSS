import torch

from architecture.centralized.cyclegan import CycleGAN
from tools.utilize import *
from metrics.metrics import mae, psnr, ssim, fid

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



__all__ = ['CycleGAN']

class NIRPSCycleGAN(CycleGAN):
    def __init__(self, config, train_loader, valid_loader, assigned_loader, 
                 device, file_path, batch_limit_weight=1.0):
        super(NIRPSCycleGAN, self).__init__(config=config, train_loader=train_loader, valid_loader=valid_loader, assigned_loader=assigned_loader, 
                 device=device, file_path=file_path, batch_limit_weight=batch_limit_weight)
     
    @torch.no_grad()
    def infer_nirps_dataset(self, save_a_path, save_b_path, gt_a_path, gt_b_path, data_loader):
        """
        src_epoch_path: source domain image path for each epoch
        tag_epoch_path: target domain image path for each epoch
        """
        assert data_loader.batch_size == 1, 'infer nirps should be for single image'
        for i, batch in enumerate(data_loader):
            imgs, _ = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, _, _, = imgs
            if i <= self.config['num_img_save']:
                mae_b = mae(real_b, fake_b).item()
                psnr_b = psnr(real_b, fake_b).item()
                ssim_b = ssim(real_b, fake_b).item()

                mae_a = mae(real_a, fake_a).item()
                psnr_a = psnr(real_a, fake_a).item()
                ssim_a = ssim(real_a, fake_a).item()

                path_a = '{}/{}-slice-{}'.format(save_a_path, batch['name_a'][0], batch['slice_num'].numpy()[0])
                path_b = '{}/{}-slice-{}'.format(save_b_path, batch['name_b'][0], batch['slice_num'].numpy()[0])
                path_a_gt = '{}/{}-slice-{}'.format(gt_a_path, batch['name_a'][0], batch['slice_num'].numpy()[0])
                path_b_gt = '{}/{}-slice-{}'.format(gt_b_path, batch['name_b'][0], batch['slice_num'].numpy()[0])

                save_image(real_a, 'gt.png', path_a_gt)
                save_image(real_b, 'gt.png', path_b_gt)
                save_image(fake_a, 'm_{:.4f}_p_{:.4f}_s_{:.4f}.png'.format(mae_a, psnr_a, ssim_a), path_a)
                save_image(fake_b, 'm_{:.4f}_p_{:.4f}_s_{:.4f}.png'.format(mae_b, psnr_b, ssim_b), path_b)
                save_image(fake_a, 'img.png'.format(mae_a, psnr_a, ssim_a), path_a)
                save_image(fake_b, 'img.png'.format(mae_b, psnr_b, ssim_b), path_b)
    