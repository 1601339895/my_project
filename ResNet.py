# -*- coding: utf-8 -*-
# File  : ResNet.py
# Author: HeLei
# Date  : 2025/8/14

import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel = bias):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel,),
            nn.BatchNorm2d(),
            nn.ReLU(),
        )

        self.project = nn.Conv2d()


    def forward(self,x):
        short_cut = x
        x = self.Conv(x)
        x = short_cut + x
        return x  

