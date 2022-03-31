import imp
from data_io.nirps import NIRPS
from torch.utils.data import DataLoader
from tools.utilize import load_metric_result, save_metric_result


if __name__ == '__main__':
    
    nirps_path = '/home/xgy/jb-wang/M-3LAB/fedmed-kaid/work_dir/centralized/ixi/Thu Mar 31 13:27:12 2022/nirps_dataset'
    regions = ['ixi']
    modalities = {'ixi': ['t2', 'pd'],
                  }
    models = ['cyclegan'] 
    epochs = [i for i in range(1, 16)]

    nirps_dataset = NIRPS(nirps_path=nirps_path, regions=regions, modalities=modalities, models=models, epochs=epochs)
    nirps_loader = DataLoader(nirps_dataset, batch_size=1, num_workers=1, shuffle=False)

    print('load nirps dataset, size:{}'.format(len(nirps_dataset)))

    for batch in nirps_loader:
        img = batch['img']
        gt = batch['gt']
        name = batch['name']

        print(name[0])

        mae = float(load_metric_result(name[0], 'mae')) 
        psnr = float(load_metric_result(name[0], 'psnr')) 
        ssim = float(load_metric_result(name[0], 'ssim')) 

        print('mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(mae, psnr, ssim))


        # TO DO

        # kaid = KAID_MODEL()

        kaid = 0
        path_kaid = name[0]
        save_metric_result(kaid, path_kaid, 'kaid')


        break