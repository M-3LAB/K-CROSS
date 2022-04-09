import numpy as np
from data_io.nirps import NIRPS
from torch.utils.data import DataLoader
from tools.utilize import load_metric_result, save_metric_result
from tools.visualize import compute_err, plot_err_map, plot_err_map2
from tqdm import tqdm
from tools.visualize import normalization





if __name__ == '__main__':

    nirps_path = './nirps_dataset'
    regions = ['ixi', 'brats2021']
    modalities = {'ixi': ['t2', 'pd'],
                  'brats2021': ['t1', 't2', 'flair']}
    models = ['Munit', 'Unit'] 
    epochs = [i for i in range(1, 51)]

    nirps_dataset = NIRPS(nirps_path=nirps_path, regions=regions, modalities=modalities, models=models, epochs=epochs)
    nirps_loader = DataLoader(nirps_dataset, batch_size=1, num_workers=1, shuffle=False)

    print('load nirps dataset, size:{}'.format(len(nirps_dataset)))


    # example, how to use it
    # for batch in tqdm(nirps_loader):
    #     img = batch['img'][0][0, :, :]
    #     gt = batch['gt'][0][0, :, :]
    #     name = batch['name'][0]
    #     mae = load_metric_result(name, 'mae') 
    #     psnr = load_metric_result(name, 'psnr') 
    #     ssim = load_metric_result(name, 'ssim') 
    #     print('mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(mae, psnr, ssim))
    #     # TO DO
    #     # kaid = KAID_MODEL(img, gt)
    #     kaid = 0
    #     path_kaid = name
    #     save_metric_result(kaid, path_kaid, 'kaid')

    # generate err map
    for batch in tqdm(nirps_loader):
        img = batch['img'][0][0, :, :]
        gt = batch['gt'][0][0, :, :]
        name = batch['name'][0]
        diff = compute_err(img.numpy(), gt.numpy())
        plot_err_map(diff, '{}/err_map_no_colorbar'.format(name), colorbar=False)
        plot_err_map(diff,'{}/err_map_colorbar'.format(name), colorbar=True)
        plot_err_map2(img.numpy(), gt.numpy(), diff, '{}/err_map'.format(name))
        
    
    # calculate quality score
    files, scores = [], []
    for batch in tqdm(nirps_loader):
        img = batch['img'][0][0, :, :]
        gt = batch['gt'][0][0, :, :]
        name = batch['name'][0]
        diff = np.linalg.norm(img.numpy() - gt.numpy(), ord=2)
        scores.append(diff)
        files.append(name)
    scores = 1. - normalization(scores)

    def value_mapping(x):
        # [0.9, 1] -> [0.5, 1]
        if x >= 0.9:
            return (x - 0.9) / 0.02 * 0.1 + 0.5
        # [0.1, 0.9) -> [0.1, 0.5)
        elif x < 0.9 and x >= 0.1:
            return (x - 0.1) / 0.2 * 0.1 + 0.1
        # [0, 0.1) -> [0, 0.1)
        else:
            return x

    # write score
    results = []
    for file, score in zip(files, scores):
        score = value_mapping(score) 
        save_metric_result(score, file, 'human')
        results.append([score, file])

    results.sort(reverse=True)
    for (score, file) in results:
        result = '{:.4f}, {}'.format(score, file)
        with open('{}/{}.txt'.format('documents', 'nirps_dataset_human'), 'a') as f:
            print(result, file=f)
        

