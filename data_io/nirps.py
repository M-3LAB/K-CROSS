import torch
import cv2
import numpy as np
import os
import sys
sys.path.append('.')

from torch.utils.data import DataLoader
from tools.utilize import load_metric_result, save_metric_result
from data_io.base import ToTensor
import torchvision.transforms as transforms

__all__ = ['NIRPS']

class NIRPS(torch.utils.data.Dataset):
    def __init__(self, nirps_path, regions=['ixi'], modalities={'ixi': ['t2']}, models=['cyclegan'], epochs=[1, 2], size=256):
        self.nirps_path = nirps_path
        self.regions = regions
        self.modalities = modalities
        self.models = models
        self.epochs = epochs
        self.size = size

        self.nirps_dataset = []
        self.load_nirps_dataset()
        self.transform = transforms.Compose([transforms.ToPILImage(), 
                                             transforms.Resize(size=self.size),
                                             ToTensor()]) 

    def load_nirps_dataset(self):
        for dataset in self.regions:
            for moda in self.modalities[dataset]:
                for model in self.models:
                    for epoch in self.epochs:
                        file_dir = '{}/{}/{}/gt'.format(self.nirps_path, dataset, moda)
                        for root, dirs, files in os.walk(file_dir): 
                            for d in dirs:
                                path_gt = '{}/{}/{}/gt/{}/gt.png'.format(self.nirps_path, dataset, moda, d)
                                path_img = '{}/{}/{}/{}/epoch_{}/{}/img.png'.format(self.nirps_path, dataset, moda, model, epoch, d)
                                self.nirps_dataset.append([path_img, path_gt]) 

        if not self.nirps_dataset:
            raise ValueError('Load Nirps Dataset Filed!')

    def __getitem__(self, index):
        # read gray scale Image
        img = cv2.imread(self.nirps_dataset[index][0], cv2.IMREAD_GRAYSCALE)
        img = self.transform(img)
        # read gray scale Image
        gt = cv2.imread(self.nirps_dataset[index][1], cv2.IMREAD_GRAYSCALE)
        gt = self.transform(gt)
        # img path
        name = self.nirps_dataset[index][0][:-8]

        return {'img': img, 'gt': gt, 'name': name}

    def __len__(self):
        return len(self.nirps_dataset)



if __name__ == '__main__':

    nirps_path = './nirps_dataset'
    regions = ['ixi', 'brats2021']
    modalities = {'ixi': ['t2', 'pd'],
                  'brats2021': ['t1', 't2', 'flair']}
    models = ['cyclegan'] 
    epochs = [i for i in range(1, 3)]

    nirps_dataset = NIRPS(nirps_path=nirps_path, regions=regions, modalities=modalities, models=models, epochs=epochs)
    nirps_loader = DataLoader(nirps_dataset, batch_size=1, num_workers=1, shuffle=False)

    print('load nirps dataset, size:{}'.format(len(nirps_dataset)))

    for batch in nirps_loader:
        img = batch['img']
        gt = batch['gt']
        name = batch['name']

        print(name[0])

        mae = load_metric_result(name[0], 'mae') 
        psnr = load_metric_result(name[0], 'psnr') 
        ssim = load_metric_result(name[0], 'ssim') 

        print('mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(mae, psnr, ssim))

        # TO DO
        # kaid = KAID_MODEL(img, gt)

        kaid = 0
        path_kaid = name[0]
        save_metric_result(kaid, path_kaid, 'kaid')

        break

