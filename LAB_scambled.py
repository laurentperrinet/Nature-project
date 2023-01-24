#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:37:06 2023

@author: mariemdiane
"""



# importing Image class from PIL package
import glob
import scipy
import skimage
import matplotlib.pyplot as plt
from scipy import fft
from skimage import data, color
import numpy as np        
import skimage
import os
import cv2
from skimage import io

# from natsort import natsorted
from tqdm import tqdm
import numpy as np
from scipy import ndimage
from PIL import Image, ImageOps
import time
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2lab
from skimage.color import lab2rgb

from skimage import color



#im = Image.open(r"/Users/mariemdiane/Documents/image_fall.png")
im = Image.open(r"/Users/mariemdiane/Desktop/ocean.png")
plt.imshow(im)

rgb_image = np.array(im).copy()

rgb_image = rgb_image[:,:,:3]

img = rgb_image[:, :530, :]

plt.imshow(img)


###### ESSAYER DE TRANSFORMER AVEC LAB


lab_img = rgb2lab(img)
lightness_img = lab_img[:, :, 0].copy()
colorA_img = lab_img[:, :, 1].copy()
colorB_img = lab_img[:, :, 2].copy()

lightness_img

plt.imshow(lightness_img, cmap='gray')

#value_img.min(), value_img.max()

#plt.imshow(hsv_img)

# On fait la texture fft sur Lightness 

def phase_shift(img, seed=1973):
    img_ = img.copy()
    img_ = (img_-.5)*2
    F2D = fft.rfft2(img_, s=None, norm=None)
    np.random.seed(seed=seed)
    phase = 2 * np.pi * np.random.rand(F2D.shape[0], F2D.shape[1])
    F_random = np.exp(1j * phase)
    Fz_scrambled = F_random * F2D
    img_scrambled = fft.irfft2(Fz_scrambled)
    img_scrambled = (img_scrambled - img_scrambled.min())/(img_scrambled.max()-img_scrambled.min())
    return img_scrambled


# Srambling luminosité
lightness_img_scrambled = phase_shift(lightness_img, seed=12346)
lightness_img_scrambled.shape

lab_img[:, :, 0] = lightness_img_scrambled
test_image = lab2rgb(lab_img)
plt.imshow(test_image)


