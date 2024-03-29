import numpy as np
from tools.utilize import concate_tensor_lists
import torch
import copy
from tools.visualize import *

"""
Alias Numpy FFT
"""

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

__all__ = ['torch_ifft','torch_fft', 'np_fft', 'np_ifft',  
           'extract_ampl', 'torch_high_pass_filter', 'torch_low_pass_filter', 
           'np_high_pass_filter', 'np_low_pass_filter']

def torch_fft(mri_img, normalized_method=None):
    """
    Convert image into K-space
    Args:
        mri_img: torch tensor (BCHW), the input channel is 1 for medical image 
    Return:
        k-space: torch tensor 
    """
    #k_space = torch.randn(mri_img.size())
    #for i in range(mri_img.size[0]): 
    #    mri_2d_img = mri_img[i, 0, :, :]
    #    k_space_2d = torch.fft.fftshift(torch.fft.fft2(mri_2d_img)) 
    #    concate_tensor_lists(k_space, k_space_2d, i)
    
    #k_space = torch.fft.fftshift(torch.fft.fft2(mri_img, norm=normalized_method)) 
    k_space = torch.fft.fft2(mri_img, norm=normalized_method) 
    return k_space

def torch_ifft(k_space, normalized_method=None):
    """
    Convert K-space into image
    Args:
        k-space: torch tensor 
    Return:
        mri_img: torch tensor (BCHW) 
    """
    #mri_img_back = torch.randn(k_space.size())
    #for i in range(k_space.size[0]):
    #    k_space_2d = k_space[i, 0, :, :]
    #    mri_2d_img = torch.fft.ifft2(torch.fft.ifftshift(k_space_2d)) 
    #    concate_tensor_lists(mri_img_back, mri_2d_img, i)

    #mri_img = torch.fft.ifft2(torch.fft.ifftshift(k_space), norm=normalized_method) 
    mri_img = torch.fft.ifft2(k_space, norm=normalized_method) 
    return mri_img

def np_fft(mri_img):
    """
    Convert image into K-space
    Args:
        mri_img: np.ndarray
    Return:
        k-space: np.ndarray
    """
    k_space = fftshift(fft2(mri_img))
    return k_space

def np_ifft(k_space):
    """
    Convert K-space into image
    Args:
        k-space: np.ndarray
    Return:
        mri_img: np.ndarray
    """
    mri_img_back = np.abs(ifft2(ifftshift(k_space))) 
    return mri_img_back

def extract_ampl(mri_img, normalized_method=None):
    """
    Convert image into K-space_abs 
    Args:
        mri_img: torch tensor (BCHW) 
    Return:
        k-space_abs: torch tensor 

    Magnitude: sqrt(re^2 + im^2) tells you the amplitude of the component 
    at the corresponding frequency
    """
    #k_space_abs = torch.randn(k_space.size())
    #for i in range(k_space_abs.size(0)):

    k_space = torch_fft(mri_img, norm=normalized_method)
    k_space_abs = torch.abs(k_space)
    return k_space_abs

def np_high_pass_filter(kspace, radius):

    """
    High pass filter removes the low spatial frequencies from k-space
    This function deletes the center of kspace by removing values
    inside a circle of given size. The circle's radius is determined by
    the 'radius' float variable (0.0 - 100) as ratio of the lenght of
    the image diagonally.
    Parameters:
        kspace (np.ndarray): Complex kspace data
        radius (float): Relative size of the kspace mask circle (percent)
    """

    if radius > 0:
        high_freq_kspace = kspace.copy()
        r = np.hypot(*high_freq_kspace.shape) / 2 * radius / 100
        rows, cols = np.array(high_freq_kspace.shape, dtype=int)
        a, b = np.floor(np.array((rows, cols)) / 2).astype(np.int)
        y, x = np.ogrid[-a:rows - a, -b:cols - b]
        mask = x * x + y * y <= r * r
        high_freq_kspace[mask] = 0
    return high_freq_kspace 

def torch_high_pass_filter(k_space, radius):
    """
    Args:
        k_space: torch tensor, BCHW 
        radius: Relative size of the kspace mask circle (percent) 
        return: high_frequency_k_space
    """
    if radius > 0:
        kspace = copy.deepcopy(k_space)
        high_freq_kspace = torch.zeros_like(kspace)
        bs, _, _, _ = kspace.size() # bs: batch_size
        for i in range(bs):
            ks = kspace[i, 0, :]
            ks_np = bchw_to_np(ks)
            hf_np = np_high_pass_filter(ks_np, radius)
            high_freq_kspace[i, 0, :] = np_to_bchw(hf_np)
        
        return high_freq_kspace
 
def np_low_pass_filter(kspace: np.ndarray, radius: float):

    """
    Low pass filter removes the high spatial frequencies from k-space
    This function only keeps the center of kspace by removing values
    outside a circle of given size. The circle's radius is determined by
    the 'radius' float variable (0.0 - 100) as ratio of the lenght of
    the image diagonally
    Parameters:
        kspace (np.ndarray): Complex kspace data
        radius (float): Relative size of the kspace mask circle (percent)
    """

    if radius < 100:
        low_freq_kspace = kspace.copy()
        r = np.hypot(*low_freq_kspace.shape) / 2 * radius / 100
        rows, cols = np.array(low_freq_kspace.shape, dtype=int)
        a, b = np.floor(np.array((rows, cols)) / 2).astype(np.int)
        y, x = np.ogrid[-a:rows - a, -b:cols - b]
        mask = x * x + y * y <= r * r
        low_freq_kspace[~mask] = 0
    return low_freq_kspace

def torch_low_pass_filter(k_space, radius):
    """
    Args:
        k_space: torch tensor, BCHW 
        radius: Relative size of the kspace mask circle (percent) 
        return: low_frequency_k_space
    """
    if radius < 100:
        kspace = copy.deepcopy(k_space)
        low_freq_kspace = torch.zeros_like(kspace)
        bs, _, _, _ = kspace.size() # bs: batch_size
        for i in range(bs):
            ks = kspace[i, 0, :]
            ks_np = bchw_to_np(ks)
            lf_np = np_low_pass_filter(ks_np, radius)
            low_freq_kspace[i, 0, :] = np_to_bchw(lf_np)
        
    return low_freq_kspace