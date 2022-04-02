import fractions
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from data_preprocess.common import *
import torch

__all__ = ['plot_sample', 'slices_reader', 'np_normalize', 'np_scaling_kspace', 'np_to_bchw',
           'bchw_to_np', 'torch_normalize', 'compute_err', 'plot_err_map', 'plot_err_map2']

def plot_sample(real_a, fake_a, real_b, fake_b, step, img_path, descript='Epoch'):
    plt.figure(figsize=(5, 4))
    plt.scatter(real_a[:400, 0], real_a[:400, 1], color='b', label='Real A', s=1, alpha=0.5)
    plt.scatter(fake_a[:400, 0], fake_a[:400, 1], color='r', label='Fake A', s=1, alpha=0.5)
    plt.scatter(real_b[:400, 0], real_b[:400, 1], color='k', label='Real B', s=1, alpha=0.5)
    plt.scatter(fake_b[:400, 0], fake_b[:400, 1], color='g', label='Fake B', s=1, alpha=0.5)

    plt.xlim((0.1, 0.4))
    plt.ylim((0.1, 0.4))
    plt.grid()
    plt.legend(loc=1)
    plt.title('{}: {}'.format(descript, step))
    plt.savefig(img_path)
    plt.close()

def slices_reader(file_path, index=74):
    mri_slices = read_img_sitk(file_path)
    mri = mri_slices[index]
    return mri

def np_normalize(f: np.ndarray):
    """ 
    Normalises array by "streching" all values to be between 0-255.
    Parameters:
        f (np.ndarray): input array
    """
    fmin = float(f.min())
    fmax = float(f.max())
    if fmax != fmin:
        coeff = fmax - fmin
        f[:] = np.floor((f[:] - fmin) / coeff * 255.)

    f = np.require(f, np.uint8)
    return f

def np_scaling_kspace(k_space):
    k_space_abs = np.abs(k_space)
    scaling = np.power(10., -2)
    np.log1p(k_space_abs * scaling, out=k_space_abs)
    np_normalize(k_space_abs)
    k_space_abs = np.require(k_space_abs, np.uint8) 
    return k_space_abs

def torch_normalize(f):
    """ 
    Normalises torch tensor by "streching" all values to be between 0-255.
    Parameters:
        f (torch tensor): BCHW, C = 1 due to the characteristics of medicial image    
    """
    for i in range(f.size()[0]):
        fmax = float(torch.max(f[i, 0, :]))
        fmin = float(torch.min(f[i, 0, :]))
        if fmax != fmin:
            coeff = fmax - fmin
            f[i,0, :] = torch.floor((f[i,0, :] - fmin) / coeff * 255.)
    f = f.to(torch.uint8)
    return f

def torch_scaling_kspace(k_space):
    k_space_abs = torch.abs(k_space)
    scaling = 0.01
    torch.log1p(scaling * k_space_abs, out=k_space_abs)
    torch_normalize(k_space_abs)
    #k_space_abs = k_space_abs.to(torch.uint8)
    return k_space_abs

def np_to_bchw(mri_img):
    mri_img = np.array(mri_img)
    mri_img = torch.from_numpy(mri_img)
    mri_img = torch.unsqueeze(torch.unsqueeze(mri_img, dim=0), dim=0) 
    return mri_img

def bchw_to_np(tensor):
    img = torch.squeeze(tensor).numpy()
    return img

def normalization(img):
    _range = np.max(img) - np.min(img)
    return (img - np.min(img)) / _range

def compute_err(img, gt):
    img_diff = 255 - np.abs(img - gt)
    # img_diff = normalization(img_diff)
    img_diff = cv2.normalize(img_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img_diff = cv2.applyColorMap(img_diff, cv2.COLORMAP_JET) 

    return img_diff

def plot_err_map(diff, img_path, colorbar=True, descript='Error Map with Colorbar'):
    if colorbar:
        plt.style.use('classic')
        plt.figure(figsize=(20, 4))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(diff)
        plt.colorbar(fraction=0.02, pad=0.05)
        plt.clim(0, 1)
        plt.savefig(img_path, bbox_inches='tight', pad_inches = 0.1)
        plt.close()
    else:
        fig = plt.figure()
        fig_ax = fig.add_axes([0, 0, 1, 1], frame_on = False)
        fig_ax.xaxis.set_visible(False)
        fig_ax.yaxis.set_visible(False)
        plt.imshow(diff)
        plt.savefig(img_path, bbox_inches='tight', pad_inches = 0)
        plt.close()


def plot_err_map2(img, gt, diff, img_path, descript='Error Map'):
    plt.style.use('classic')
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('GT')

    plt.subplot(1, 3, 2)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Fake')

    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.xticks([])
    plt.yticks([])
    plt.title('Error Map')

    # plt.colorbar(fraction=0.02, pad=0.05)
    # plt.clim(0, 1)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
 
if __name__ == '__main__':

    img = cv2.imread('./work_dir/centralized/brats2021/Sat Mar 26 00:30:35 2022/images/epoch_15/BraTS2021_00114-slice-67/fake_a.png', cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread('./work_dir/centralized/brats2021/Sat Mar 26 00:30:35 2022/images/epoch_15/BraTS2021_00114-slice-67/real_a.png', cv2.IMREAD_GRAYSCALE)
    img_path = './tools/err_map.png'
    diff = compute_err(img, gt)
    plot_err_map(diff, img_path)
