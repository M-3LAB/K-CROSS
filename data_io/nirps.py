import torch
import cv2
import os

from torch.utils.data import DataLoader

__all__ = ['NIRPS']

class NIRPS(torch.utils.data.Dataset):
    def __init__(self, nirps_path, regions=['ixi'], modalities={'ixi': ['t1']}, models=['cyclegan'], epochs=[1, 2]):
        self.nirps_path = nirps_path
        self.region = regions
        self.modalities = modalities
        self.models = models
        self.epochs = epochs

        self.nirps_dataset = []
        self.load_nirps_dataset()

    def load_nirps_dataset(self):
        for dataset in self.region:
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
        img = cv2.imread(self.nirps_dataset[index][0])
        gt = cv2.imread(self.nirps_dataset[index][1])
        name = self.nirps_dataset[index][0]

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

        print(name)
        break

