# -*- coding: utf-8 -*-
# File  : data_loader.py
# Author: HeLei
# Date  : 2025/4/1

import torchvision.transforms as transforms
# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class RainDataset(Dataset):
    def __init__(self, cfg):
        self.input_dir = os.path.join(cfg.data_root, "input")
        self.gt_dir = os.path.join(cfg.data_root, "gt")
        self.image_list = os.listdir(self.input_dir)
        self.transform = transforms.Compose([
            transforms.RandomCrop(cfg.image_size),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        input_img = Image.open(os.path.join(self.input_dir, self.image_list[idx]))
        gt_img = Image.open(os.path.join(self.gt_dir, self.image_list[idx]))
        return self.transform(input_img), self.transform(gt_img)


def get_loader(cfg):
    dataset = RainDataset(cfg)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
