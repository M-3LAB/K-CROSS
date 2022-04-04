import cv2
import numpy as np
import os

from data_io.nirps import NIRPS
from torch.utils.data import DataLoader
from tools.utilize import load_metric_result, save_metric_result
from tqdm import tqdm




def articial_scoring_system():
    '''
    Press the score twice to score successfully, which can be modified repeatedly. 
    Press enter to save the result to artificial.txt. 
    If you don't mark, press enter directly, and the original score will be saved into artificial.txt  
    Press ESC, the system will exit.
    '''  
    nirps_path = './nirps_dataset'
    regions = ['ixi', 'brats2021']
    modalities = {'ixi': ['t2', 'pd'],
                  'brats2021': ['t1', 't2', 'flair']}
    models = ['cyclegan'] 
    epochs = [i for i in range(1, 51)]

    nirps_dataset = NIRPS(nirps_path=nirps_path, regions=regions, modalities=modalities, models=models, epochs=epochs)
    nirps_loader = DataLoader(nirps_dataset, batch_size=1, num_workers=1, shuffle=False)

    print('load nirps dataset, size:{}'.format(len(nirps_dataset)))

    # clear all labeled data
    # for batch in nirps_loader:
    #     name = batch['name'][0]

    #     if os.path.exists(name + '/artificial.txt'):
    #         os.remove(name + '/artificial.txt')

    # return

    for i, batch in enumerate(nirps_loader):
        print('-------- {} / {} ---------'.format(i, len(nirps_dataset)))
        name = batch['name'][0]

        kaid = load_metric_result(name, 'human')
        img = cv2.imread(name + '/err_map.png')
        cv2.namedWindow('Artificial-Score-System: {}'.format(name), cv2.WINDOW_AUTOSIZE)
        cv2.putText(img, 'src: {}'.format(int(kaid * 10) + 1), (12,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3) 

        if os.path.exists(name + '/artificial.txt'):
            arti = load_metric_result(name, 'artificial') 
            cv2.putText(img, 'saved: {}'.format(int(arti * 10)), (12, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3) 
        else:
            save_metric_result(int(kaid * 10 + 1) / 10., name, 'artificial')


        cv2.imshow('Artificial-Score-System: {}'.format(name), img)

        # press ENTER to continue
        while(cv2.waitKey(0) != 13):
            # wait keyboard [1, 2, 3, ..., 0]
            s = cv2.waitKey(0)
            # press ESC, return
            if s == 27:
                return 

            s =  int(s) - 48
            print(s)
            cv2.putText(img, 'change to: {}'.format(s), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3) 
            cv2.imshow('Artificial-Score-System: {}'.format(name), img)
            save_metric_result(s / 10., name, 'artificial')

        cv2.destroyAllWindows()



if __name__ == '__main__':
    articial_scoring_system()