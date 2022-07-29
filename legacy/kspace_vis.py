from tkinter import W
from model.kaid.complex_nn.fourier_transform import np_fft
import numpy as np
from tools.visualize import *
import matplotlib.pyplot as plt
from tools.utilize import *
from model.kaid.complex_nn.fourier_transform import * 
import cv2
import os

for i in range(10):
    path = os.path.join('/Users/guoyang/Downloads/eval', str(i))
    img_path = os.path.join(path, 'img.png') 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
    k_space = np_fft(img)
    #k_space_abs = np_scaling_kspace(k_space)
    k_space_abs = np.abs(k_space)
    k_space_angle = np.angle(k_space)
    k_space_abs_back = np_ifft(k_space_abs)
    k_space_angle_back = np_ifft(k_space_angle)
    ks_path = os.path.join(path, 'ks.png')
    ks_abs_back_path = os.path.join(path, 'ks_abs_back.png')
    ks_angle_back_path = os.path.join(path, 'ks_angle_back.png')
    #cv2.imwrite(ks_path, k_space_abs)
    cv2.imwrite(ks_abs_back_path, k_space_abs_back)
    cv2.imwrite(ks_angle_back_path, k_space_angle_back)

