import cv2
import numpy as np
import PIL

from data_io.nirps import NIRPS
from torch.utils.data import DataLoader
from tools.utilize import load_metric_result, save_metric_result
from tqdm import tqdm






if __name__ == '__main__':
    
    nirps_path = './nirps_dataset'
    regions = ['ixi', 'brats2021']
    modalities = {'ixi': ['t2', 'pd'],
                  'brats2021': ['t1', 't2', 'flair']}
    models = ['cyclegan'] 
    epochs = [i for i in range(1, 51)]

    nirps_dataset = NIRPS(nirps_path=nirps_path, regions=regions, modalities=modalities, models=models, epochs=epochs)
    nirps_loader = DataLoader(nirps_dataset, batch_size=1, num_workers=1, shuffle=False)

    print('load nirps dataset, size:{}'.format(len(nirps_dataset)))

    for batch in tqdm(nirps_loader):
        name = batch['name'][0]

        kaid = load_metric_result(name, 'kaid')
        img = PIL.Image.open(name + '/err_map.png')
        img.show(0)
        # img = cv2.imread(name + '/err_map.png')
        # cv2.namedWindow('Hello', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Hello', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pass
