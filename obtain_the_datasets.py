import os

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from autoattack import AutoAttack

from ddim import ddim_denoising
from gaussion_noise import add_gaussion_noise
from lennet import LeNet
from pepper_salt_noise import add_salt_and_pepper_noise
from student_image_reconstraction_function import student_image_editing_sample
from teacher_image_reconstraction_function import teacher_image_editing_sample

device = torch.device('cuda:0')

store_path = 'lenets_fashionmnist.pt'
lenet = LeNet()
ckpt = torch.load(
        store_path, map_location=device)
lenet.load_state_dict(ckpt)
print('model loaded')
lenet.to(device)
lenet.eval()

# 加载FashionMNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2048,
                                          shuffle=False, num_workers=2)
dataiter = iter(trainloader)
images, labels = next(dataiter)
images = images.cuda()
labels = labels.cuda()
#save_image(images*-1+1, os.path.join(
#            "./student_noise/",  "Guidence_Fashionmnist_images120step.bmp"), nrow=8)
#adversary = AutoAttack(lenet, norm='L2', eps=0.5, version='standard')
# 添加椒盐噪声
with torch.no_grad():
 noise_level = 0.2  # 设置噪声水平
 #noisy_images_gs = add_gaussion_noise(images, mean=0, std=0.2).to(device) fasionmnist要隐藏
 #noisy_images_sp = add_salt_and_pepper_noise(images, noise_level).to(device) fashionmnist要隐藏
 #noisy_images_at = adversary.run_standard_evaluation(images, labels, bs=64).to(device)


 #save_image(noisy_images*-1+1, os.path.join(
 #           "./student_noise/",  "w2.0_noisyimages_150step_level0.2.bmp"), nrow=8)
 #reimage_gs = ddim_denoising(noisy_images_gs,  150).to(device) fashinmnist要隐藏
 #reimage_sp = ddim_denoising(noisy_images_sp,  150).to(device)  fashionmnist要隐藏
 #reimage_at = ddim_denoising(noisy_images_at,  150).to(device)
 #save_image(reimage*-1+1, os.path.join(
 #           "./student_noise/",  "student_w2.0_reimage_150step_level0.2.bmp"), nrow=8)
 print('already saved')

transform_clas = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
transform_at = transforms.Compose([
            transforms.Resize((28, 28)),
        ])

adversary = AutoAttack(lenet, norm='L2', eps=0.5, version='standard')
image_clas = transform_clas(images)

noisy_images_gs = add_gaussion_noise(image_clas, mean=0, std=0.2).to(device)
noisy_images_sp = add_salt_and_pepper_noise(image_clas, noise_level).to(device)
noisy_images_at = adversary.run_standard_evaluation(image_clas, labels, bs=64).to(device)
#noisy_images_at_1 = transform_at(noisy_images_at)

reimage_gs = ddim_denoising(noisy_images_gs, 50).to(device)
reimage_sp = ddim_denoising(noisy_images_sp, 50).to(device)
reimage_at = ddim_denoising(noisy_images_at, 50).to(device)


reimage_clas_gs = transform_clas(reimage_gs)
reimage_clas_sp = transform_clas(reimage_sp)
reimage_clas_at = transform_clas(reimage_at)
adversary1 = AutoAttack(lenet, norm='L2', eps=0, version='apgd-ce')
ans_images_gs = adversary1.run_standard_evaluation(reimage_clas_gs, labels, bs=64)
print('gs finish')
ans_images_sp = adversary1.run_standard_evaluation(reimage_clas_sp, labels, bs=64)
print('sp finish')
ans_images_at = adversary1.run_standard_evaluation(reimage_clas_at, labels, bs=64)
print('at finish')