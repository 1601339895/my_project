# -*- coding: utf-8 -*-
# File  : model.py
# Author: HeLei
# Date  : 2025/4/1

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class PhysicalConstraint(nn.Module):
    """物理约束模块：通过梯度域约束实现平滑先验"""

    def forward(self, pred):
        grad_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_y = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return (grad_x.abs().mean() + grad_y.abs().mean())


class ComplexCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 128)  # 128x128x128
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
        self.constraint = PhysicalConstraint()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        mse_loss = nn.MSELoss()
        mse_loss = mse_loss(pred, target)
        constraint_loss = self.constraint(pred)
        return mse_loss + constraint_loss    # 加上约束项
