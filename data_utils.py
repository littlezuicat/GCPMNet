"""
Author: Yi Ren
Email: renyi@buu.edu.cn
Affiliation: Beijing Union University, Beijing, China
"""
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from torch.nn import functional as FF_nn
import os,sys, re
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
from tqdm import tqdm
import json
from models.prompt import TextPromptEncoder
from models.Transformer import SwinT_KernelGenerator
from tqdm import tqdm

BS=opt.bs
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

class HSVDataset(data.Dataset):
    def __init__(self, path, train, size=crop_size):
        super().__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.haze_imgs = []
        self.clear_imgs = []
        self.kernels = []
        #嵌入模型
        self.prompt_encoder = TextPromptEncoder().to(opt.device)
        self.kernel_generator = SwinT_KernelGenerator().to(opt.device)
        name = opt.dataset_name.lower().replace('-', '').replace('_', '')
        if any(haze in name for haze in ['ohaz', 'nhhaz', 'ihaz', 'densehaz']):
            
            hazy_dir = os.path.join(path, 'hazy')
            clear_dir = os.path.join(path, 'GT')
            json_path = os.path.join(path, 'prompts.json') # 清单文件路径

            # 读取 JSON 清单文件
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 遍历清单，构建文件路径列表和提示词列表
            with torch.no_grad():
                for item in tqdm(metadata):
                    hazy_img_path = os.path.join(hazy_dir, item['hazy_file'])
                    clear_img_path = os.path.join(clear_dir, item['gt_file'])
                    
                    hazy_img = Image.open(hazy_img_path).convert("RGB")
                    clear_img = Image.open(clear_img_path).convert("RGB")
                    self.haze_imgs.append(hazy_img)
                    self.clear_imgs.append(clear_img)
                    kernel = self.prompt_encoder(item['prompt']) + self.kernel_generator(tfs.ToTensor()(hazy_img))
                    self.kernels.append(kernel)
                   
        else:
            print('无法使用数据集, 仅支持O-Haze、NHHaze、IHaze或DenseHaze数据集, 请确保数据集命名正确并包含prompts.json文件。')
        print(f"[Dataset loaded successfully]\nLoaded {len(self.haze_imgs)} hazy images and {len(self.clear_imgs)} GT images into memory.\nNote: Loading the entire dataset into memory can significantly speed up training; high memory usage is expected.")

    def __getitem__(self, index):
        haze = self.haze_imgs[index]  # Directly retrieved from memory
        clear = self.clear_imgs[index]  # Directly retrieved from memory
        kernel = self.kernels[index]  # Directly retrieved from memory
        # print(haze)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze, clear)
        return haze, kernel, clear

    def augData(self, data, target):
        
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        
            
        if not self.train:
            shaper = tfs.Resize((512, 512))
            data = shaper(data)
            target = shaper(target)
        data = tfs.ToTensor()(data)
        # data = tfs.Normalize(mean=mean, std=std)(data)
        target = tfs.ToTensor()(target)
        
            
        return data, target

    def __len__(self):
        return len(self.haze_imgs)



path=opt.path

loader_train=DataLoader(dataset=HSVDataset(path+f'/{opt.dataset_name}/train',train=True,size=crop_size),batch_size=opt.bs,shuffle=True, num_workers=opt.workers)
loader_test=DataLoader(dataset=HSVDataset(path+f'/{opt.dataset_name}/test',train=False,size='whole image'),batch_size=1,shuffle=False, num_workers=opt.workers)



if __name__ == "__main__":
    pass
