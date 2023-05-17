
from diffusers import DDIMPipeline, DDPMPipeline
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from PIL import Image
import torch
import os
import numpy as np
from torchvision.utils import save_image
'''
#model_id = "nabdan/mnist_20_epoch"
model_id = "ynwag9/fashion_mnist_ddpm_32"
# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id)
scheduler = ddim.scheduler
model = ddim.unet
model.to("cuda")
scheduler.set_timesteps(100)

sample_size = model.config.sample_size
print(sample_size)
# run pipeline in inference (sample random noise and denoise)
#image = ddpm(num_inference_steps=1000).images[0]

# save image
#image.save("ddim_generated_fashion_image.bmp")
noise = torch.randn((80, 1, sample_size, sample_size)).to("cuda")
input = noise

for t in scheduler.timesteps:
    print(t)
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
print(image.shape)
save_image(image*-1+1, os.path.join(
           "./student_noise/",  "666.bmp"), nrow=8)
'''

'''
def ddim_denoising(noisy_images, step):
 
 scheduler = DDPMScheduler.from_pretrained("sketch2img-FashionMNIST/scheduler")
 model = UNet2DModel.from_pretrained("ksaml/mnist-fashion_64").to("cuda")
 scheduler.set_timesteps(step)

 sample_size = model.config.sample_size
 print(sample_size)
 #noise = torch.randn((80, 1, sample_size, sample_size)).to("cuda")
 input = noisy_images

 for t in scheduler.timesteps:
    print(t)
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

 image = (input / 2 + 0.5).clamp(0, 1)
 print(image.shape)
 return image
#image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
#image = Image.fromarray((image * 255).round().astype("uint8"))
'''
def ddim_denoising(noisy_images, step):
    model_id = "ynwag9/fashion_mnist_ddpm_32"
    # load model and scheduler
    ddim = DDIMPipeline.from_pretrained(model_id)
    scheduler = ddim.scheduler
    model = ddim.unet
    model.to("cuda")
    scheduler.set_timesteps(step)

    sample_size = model.config.sample_size
    print(sample_size)
    # run pipeline in inference (sample random noise and denoise)
    # image = ddpm(num_inference_steps=1000).images[0]

    # save image
    # image.save("ddim_generated_fashion_image.bmp")
    #noise = torch.randn((80, 1, sample_size, sample_size)).to("cuda")
    input = noisy_images

    for t in scheduler.timesteps:
        print(t)
        with torch.no_grad():
            noisy_residual = model(input, t).sample
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
            input = prev_noisy_sample

    image = (input / 2 + 0.5).clamp(0, 1)
    print(image.shape)
    return image
