import numpy as np
import os
from tools.utilize import load_metric_result

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import matplotlib
matplotlib.use('Agg')

__all__ = ['MetricConsistency']

class MetricConsistency():
    def __init__(self, nirps_path, region='ixi', modality='t2', model='cyclegan', epochs=[1, 2], file_names=None, metrics=['psnr']):
        self.nirps_path = nirps_path
        self.region = region
        self.modality = modality
        self.model = model
        self.epochs = epochs
        self.file_names = file_names
        self.metrics = metrics

        self.metric_src_results = []
        self.metric_std_results = []
        self.load_src_results()
        self.get_std_results()

    def load_src_results(self):
        if not self.file_names:
            file_dir = '{}/{}/{}/gt'.format(self.nirps_path, self.region, self.modality)
            for root, dirs, files in os.walk(file_dir):
                self.file_names = dirs
                break

        for file_name in self.file_names:
            file_results = []
            for metric in self.metrics:
                metric_results = []
                for epoch in self.epochs:
                    path = '{}/{}/{}/{}/epoch_{}/{}'.format(self.nirps_path, self.region, self.modality, self.model, epoch, file_name)
                    result =  load_metric_result(path, metric) 
                    metric_results.append(result)        
                file_results.append(metric_results)
            self.metric_src_results.append(file_results)

        if not self.metric_src_results:
            raise ValueError('Load Metric Result Filed!')

    def transform_src_to_std(self, x, reverse=False, type_int=False):
        _range = np.max(x) - np.min(x)
        std = (x - np.min(x)) * 1 / _range
        if reverse:
            std = 1. - std
        if type_int:
            std = std.astype(np.int8)
        
        return list(std)

    def get_std_results(self):
        for file in self.metric_src_results:
            std_result = []
            for indicator, value in zip(self.metrics, file):
                if self.metrics[indicator] == 'down':
                    std_result.append(self.transform_src_to_std(value, reverse=True, type_int=False))
                else:
                    std_result.append(self.transform_src_to_std(value, reverse=False, type_int=False))
            self.metric_std_results.append(std_result)

def vis_metric_consistency(file):
    
    plt.figure(figsize=(6, 4))

    xtick = len(file[0])
    xtick = 20
    x = range(1, xtick+1, 1)

    plt.plot(x, np.abs(np.array(file[4])[:xtick] - np.array(file[3])[:xtick]), color='red', linewidth=1, alpha=1, linestyle='-', marker='.', label='KAID')
    plt.plot(x, np.abs(np.array(file[0])[:xtick] - np.array(file[3])[:xtick]), color='black', linewidth=1, marker='.', linestyle='-', label='MAE')
    plt.plot(x, np.abs(np.array(file[1])[:xtick] - np.array(file[3])[:xtick]), color='blue', linewidth=1, alpha=1, linestyle='-', marker='.', label='PSNR')
    plt.plot(x, np.abs(np.array(file[2])[:xtick] - np.array(file[3])[:xtick]), color='green', linewidth=1, alpha=1, linestyle='-', marker='.', label='SSIM')
    # plt.plot(x, file[3] - file[3], color='black', linewidth=1, alpha=1, linestyle='-', marker='.', label='Radiologist')


    plt.xlim(0, xtick)
    plt.ylim(0, 1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # plt.title('Discrepancy between Metric and Radiologist Score', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    # plt.legend(loc='lower right',fontsize='medium')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.savefig('./documents/metric_consistency.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':

    nirps_path = './nirps_dataset'
    # only load one dataset
    region = 'ixi' # 'brats2021'
    region = 'brats2021' # 'brats2021'
    # only load one modality
    modality = 't2'
    # only load one model
    model = 'unit'
    # should assign epoch range
    epochs = [i for i in range(1, 41)]
    # should assign file name; when file_name = None, load all files
    # file_name = None
    file_name = ['BraTS2021_01767-slice-76']
    # should assign metric, add kaid
    # metrics = {'mae': 'down', 'psnr': 'up', 'ssim': 'up', 'artificial': 'up', 'kaid_combined': 'down'}
    metrics = {'mae': 'down', 'psnr': 'up', 'ssim': 'up', 'artificial': 'up', 'kaid_complex': 'down'}
    metrics = {'mae': 'down', 'psnr': 'up', 'ssim': 'up', 'artificial': 'up', 'kaid_normal': 'down'}

    metric_dataset = MetricConsistency(nirps_path=nirps_path, region=region, modality=modality, epochs=epochs, file_names=file_name, metrics=metrics)
    # [file_name, metric, epoch]
    result_src = metric_dataset.metric_src_results
    result_std = metric_dataset.metric_std_results

    # plot result
    vis_metric_consistency(result_std[0])

