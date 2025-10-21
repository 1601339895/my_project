# -*- coding: utf-8 -*-
# File  : test.py
# Author: HeLei
# Date  : 2025/4/1
import os

# test.py
import torch
from model import CNNModel
from PINN_Image_Restoration.config import Config
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image

cfg = Config()
model = CNNModel().to(cfg.device)
model.load_state_dict(torch.load(f"{cfg.save_dir}/best.pth"))


def evaluate(input_path, output_path):
    input_img = Image.open(input_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor()])

    with torch.no_grad():
        input_tensor = transform(input_img).unsqueeze(0).to(cfg.device)
        output, _ = model(input_tensor, calc_phys=False)
        save_image(output, output_path)
        return psnr(input_tensor.cpu().numpy(), output.cpu().numpy())


# 批量测试代码
test_inputs = os.listdir("./test_data/input")
for img_name in test_inputs:
    psnr_value = evaluate(f"./test_data/input/{img_name}",
                          f"./results/{img_name}")
    print(f"PSNR: {psnr_value:.2f}dB")
