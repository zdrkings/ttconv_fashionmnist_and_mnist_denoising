
import os
import random

import numpy as np

import torch
import torchvision.utils as tvu

from Model_teacher import Teacher_ContextUnet
from ddpm_midstep_denoising import midstep_denoising
from ddpm_noising import Append_Gaussion_noise
torch.manual_seed(124)
device = torch.device('cuda:0')
store_path_teacher = 'model_mnist_cnn_19.pt'
midstep_denoising_teacher = midstep_denoising(nn_model=Teacher_ContextUnet(in_channels=1, n_feat=256, n_classes=10), betas=(1e-4, 0.02), n_T=500,
                device=device, drop_prob=0.1)
ckpt = torch.load(
        store_path_teacher, map_location=device)
midstep_denoising_teacher.load_state_dict(ckpt)
midstep_denoising_teacher.eval()
'''
print("Model's state_dict:")
for param_tensor in midstep_denoising1.state_dict():
    print(param_tensor, "\t", midstep_denoising1.state_dict()[param_tensor].size())
print('model_loaded')
'''

def teacher_image_editing_sample(img, labels,steps):
    assert isinstance(img, torch.Tensor)
    with torch.no_grad():

        assert img.ndim == 4, img.ndim
        x0 = img
        #xs = []
        #for it in range(2): #这个参数也可以进行修改

        total_noise_levels = steps #可以修改加入噪声的步数
        Append_Gaussion_noise1 = Append_Gaussion_noise(betas=(1e-4, 0.02), n_T=500) #先实例化一个
        Append_Gaussion_noise1.to(device)
        x_t_upto= Append_Gaussion_noise1(x0, total_noise_levels)
        #midstep_denoising1 = midstep_denoising(nn_model=Teacher_ContextUnet(in_channels=1, n_feat=256, n_classes=10), betas=(1e-4, 0.02), n_T=500,
        #        device=device, drop_prob=0.1)
        x_reconstraction = midstep_denoising_teacher.resample(x_t_upto, labels,total_noise_levels,(1,28,28),device,2.0)



        x0 = x_reconstraction
            #xs.append(x0)

        #return torch.cat(xs, dim=0)
        return x0
