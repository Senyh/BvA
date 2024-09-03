import os
import SimpleITK as sitk
import numpy as np
from skimage import measure
import torch
import torch.nn as nn
from torch.fft import fftn, ifftn, fftshift, ifftshift
from medpy import metric
import math
import torch.nn.functional as F


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_image(file_name):
    image_itk = sitk.ReadImage(file_name)
    origin = image_itk.GetOrigin()
    spacing = image_itk.GetSpacing()
    image = sitk.GetArrayFromImage(image_itk)
    return image, origin, spacing


def save_image(file_name, image_array, origin, spacing):
    image_itk = sitk.GetImageFromArray(image_array)
    image_itk.SetOrigin(origin)
    image_itk.SetSpacing(spacing)
    sitk.WriteImage(image_itk, file_name)


def measure_img(o_img, t_num=1):
    p_img=np.zeros_like(o_img)
    testa1 = measure.label(o_img.astype("bool"))
    props = measure.regionprops(testa1)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    for i in range(0, t_num):
        index = numPix.index(max(numPix)) + 1
        p_img[testa1 == index]=o_img[testa1 == index]
        numPix[index-1]=0
    return p_img


def frequency_mixup(tensor, aux_tensor, device='cuda'):
    t_fft = fftn(tensor, dim=[-3, -2, -1], norm='ortho')
    t_fft = fftshift(t_fft)
    t_amp, t_pha = torch.abs(t_fft), torch.angle(t_fft)

    x_fft = fftn(aux_tensor, dim=[-3, -2, -1], norm='ortho')
    x_fft = fftshift(x_fft)
    x_amp, x_pha = torch.abs(x_fft), torch.angle(x_fft)
    
    alpha = torch.rand(1).item()
    beta = torch.rand(1).item() * 0.15
    mask = torch.zeros_like(tensor).to(device)
    c_d, c_h, c_w = tensor.shape[2] // 2, tensor.shape[3] // 2, tensor.shape[4] // 2
    length = int(beta * tensor.shape[2]) // 2
    mask[:, :, c_d-length:c_d+length, c_h-length:c_h+length, c_w-length:c_w+length] = 1

    mix_amp = t_amp * (1 - mask) * (1 - alpha) + x_amp * mask * alpha


    mix_fft = torch.polar(mix_amp, t_pha)
    mix_fft = ifftshift(mix_fft)

    mix_ifft = ifftn(mix_fft, dim=[-3, -2, -1], norm='ortho').real
    mix_ifft = torch.clamp(mix_ifft, 0, 1)
    return mix_ifft
