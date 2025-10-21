# -*- coding: utf-8 -*-
# File  : Config.py
# Author: HeLei
# Date  : 2025/4/1
# config.py

import torch

class Config:
    def __init__(self):
        self.batch_size = 16
        self.lr = 1e-4
        self.epochs = 100
        self.image_size = 256
        self.model_type = "CNN"  # 可选CNN/ViT
        self.data_root = "./data"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = "./checkpoints"
        self.phys_weight = 0.1  # 物理约束项的权重系数

        # 增加错误处理机制，检查配置项的合理性和可用性
        self.validate_config()

    def validate_config(self):
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        if not isinstance(self.lr, float) or self.lr <= 0:
            raise ValueError("lr 必须是正浮点数")
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs 必须是正整数")
        if not isinstance(self.image_size, int) or self.image_size <= 0:
            raise ValueError("image_size 必须是正整数")
        if self.model_type not in ["CNN", "ViT"]:
            raise ValueError("model_type 必须是 'CNN' 或 'ViT'")
        if not isinstance(self.phys_weight, float) or self.phys_weight < 0:
            raise ValueError("phys_weight 必须是非负浮点数")

        # 检查数据根目录是否存在
        import os
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"数据根目录 {self.data_root} 不存在")

        # 检查保存目录是否存在，如果不存在则创建
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

# 使用示例
try:
    config = Config()
    print("配置验证通过")
except (ValueError, FileNotFoundError) as e:
    print(f"配置错误: {e}")
