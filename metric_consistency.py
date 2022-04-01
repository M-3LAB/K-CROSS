import numpy as np
import os

from tools.utilize import load_metric_result

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

        self.metric_results = []
        self.load_results()

    def load_results(self):
        if not self.file_names:
            file_dir = '{}/{}/{}/gt'.format(self.nirps_path, self.region, self.modality)
            for root, dirs, files in os.walk(file_dir):
                self.file_names = dirs

        for file_name in self.file_names:
            file_results = []
            for metric in self.metrics:
                metric_results = []
                for epoch in self.epochs:
                    path = '{}/{}/{}/{}/epoch_{}/{}'.format(self.nirps_path, self.region, self.modality, self.model, epoch, file_name)
                    result =  load_metric_result(path, metric) 
                    metric_results.append(result)        
                file_results.append(metric_results)
            self.metric_results.append(file_results)

        if not self.metric_results:
            raise ValueError('Load Metric Result Filed!')

if __name__ == '__main__':

    nirps_path = './nirps_dataset'
    # only load one dataset
    region = 'ixi' # 'brats2021'
    # only load one modality
    modality = 't2'
    # only load one model
    model = 'cyclegan'
    # should assign epoch range
    epochs = [i for i in range(1, 51)]
    # if file_name = None, load all files
    file_name = ['IXI086-Guys-0728-slice-60']
    # should assign metric
    metrics = ['mae', 'psnr', 'ssim', 'kaid']

    metric_dataset = MetricConsistency(nirps_path=nirps_path, region=region, modality=modality, epochs=epochs, file_names=file_name, metrics=metrics)
    # [file_name, metric, epoch]
    result = metric_dataset.metric_results

    pass

