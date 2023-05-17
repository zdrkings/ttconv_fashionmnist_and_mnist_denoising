import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from Model_teacher import Teacher_DDPM, Teacher_ContextUnet
from MyDataset import MyDataset
from lennet import LeNet

device = torch.device('cuda:0')
store_path_teacher = 'model_teacher_fashion_19.pt'
save_dir = './teacher_datasets/'
n_epoch = 20
#batch_size = 256
n_T = 500  # 500
n_classes = 10
n_feat = 256  # 128 ok, 256 better (but slower)


ws_test = [0.0, 0.5, 1, 1.5, 2.0]  # strength of generative guidance
w=2.0

best_teacher_ddpm = Teacher_DDPM(nn_model=Teacher_ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
ckpt = torch.load(
        store_path_teacher, map_location=device)
best_teacher_ddpm.load_state_dict(ckpt)
best_teacher_ddpm.eval()
k=10
with torch.no_grad():
    n_sample = k * n_classes
    #for w_i, w in enumerate(ws_test):
    x_gen, x_gen_store = best_teacher_ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)
