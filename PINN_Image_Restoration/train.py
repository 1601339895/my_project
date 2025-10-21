# -*- coding: utf-8 -*-
# File  : train.py.py
# Author: HeLei
# Date  : 2025/4/1

import torch
from model import ComplexCNNModel
from data_loader import get_loader
from PINN_Image_Restoration.config import Config
from torch import optim, nn
import time
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter


def train(cfg):
    # 初始化模型
    model = ComplexCNNModel()

    # 如果可用，使用多GPU
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    model = model.to(cfg.device)

    # 初始化优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    # 获取数据加载器
    train_loader = get_loader(cfg)

    # 初始化最佳损失值
    best_loss = float('inf')

    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # 开始训练
    for epoch in range(cfg.epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        running_data_loss = 0.0
        running_phys_loss = 0.0

        for batch_idx, (inputs, gts) in enumerate(train_loader):
            inputs, gts = inputs.to(cfg.device), gts.to(cfg.device)
            optimizer.zero_grad()

            outputs, phys_loss = model(inputs)
            data_loss = criterion(outputs, gts)
            total_loss = data_loss + cfg.phys_weight * phys_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_data_loss += data_loss.item()
            running_phys_loss += phys_loss.item()

            if (batch_idx + 1) % cfg.log_interval == 0:
                print(f"Epoch [{epoch + 1}/{cfg.epochs}], Step [{batch_idx + 1}/{len(train_loader)}], "
                      f"Total Loss: {running_loss / cfg.log_interval:.4f}, "
                      f"Data Loss: {running_data_loss / cfg.log_interval:.4f}, "
                      f"Phys Loss: {running_phys_loss / cfg.log_interval:.4f}")
                running_loss = 0.0
                running_data_loss = 0.0
                running_phys_loss = 0.0

        # 计算并打印每个epoch的平均损失
        epoch_loss = total_loss.item() / len(train_loader)
        print(f"Epoch [{epoch + 1}/{cfg.epochs}] completed in {time.time() - epoch_start_time:.2f}s, "
              f"Avg Total Loss: {epoch_loss:.4f}, Avg Data Loss: {data_loss.item():.4f}, Avg Phys Loss: {phys_loss.item():.4f}")

        # 记录到TensorBoard
        writer.add_scalar('Loss/Total', epoch_loss, epoch)
        writer.add_scalar('Loss/Data', data_loss.item(), epoch)
        writer.add_scalar('Loss/Physical', phys_loss.item(), epoch)

        # 保存模型
        torch.save(model.state_dict(), f"{cfg.save_dir}/latest.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"{cfg.save_dir}/best.pth")

    # 关闭TensorBoard writer
    writer.close()


def main():
    # 初始化配置
    cfg = Config()

    # 调用训练函数
    train(cfg)


if __name__ == "__main__":
    main()
