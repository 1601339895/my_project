# -*- coding: utf-8 -*-
# File  : Encoder_test.py
# Author: HeLei
# Date  : 2025/4/17
## Encoder Block
from typing import List

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from LumiSenseMoE.src.net.basenet import LayerNorm, FeedForward, Attention


class PreNorm(nn.Module):
    """
    在执行某个函数 fn 之前，对输入 x 进行LayerNorm归一化处理。
    """

    def __init__(self, dim, fn):
        """
        参数 dim 是输入的维度，fn 是要执行的函数。
        :param dim:
        :param fn:
        """
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class AttentionFusionModule(nn.Module):
    """
    注意力融合模块，用于动态融合光照特征和主干特征。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)  # Query
        self.key_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)    # Key
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)       # Value
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, main_feature, light_feature):
        B, C, H, W = main_feature.shape

        # 计算Query、Key和Value
        query = self.query_conv(main_feature).view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C')
        key = self.key_conv(light_feature).view(B, -1, H * W)                       # (B, C', N)
        value = self.value_conv(light_feature).view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C)

        # 计算注意力分数
        energy = torch.bmm(query, key)  # (B, N, N)
        attention = self.softmax(energy)  # (B, N, N)

        # 加权求和
        out = torch.bmm(attention, value)  # (B, N, C)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)  # (B, C, H, W)

        # 融合主干特征和光照特征
        fused_feature = main_feature + out
        return fused_feature


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type)
        ])

        self.mixer = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        # 添加注意力融合模块
        self.attention_fusion = AttentionFusionModule(dim)

    def forward(self, x, illu_fea):
        # 动态融合主干特征和光照特征
        x = self.attention_fusion(x, illu_fea)

        # 主干处理
        x = x + self.mixer(self.norms[0](x))
        x = x + self.ffn(self.norms[1](x))
        return x


class EncoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool,n_fea_in: int):
        super().__init__()

        self.loss = None
        self.num_blocks = num_blocks

        # 光照感知模块
        self.illumination_estimator = Illumination_Estimator(dim=dim, n_fea_in=n_fea_in)

        # 编码器残差块列表
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                EncoderBlock(dim, num_heads[i], ffn_expansion, bias, LayerNorm_type)
            )

    def forward(self, x):
        # 使用光照感知模块提取光照特征
        illu_fea, illu_map = self.illumination_estimator(x)

        # 依次通过编码器残差块
        for layer in self.layers:
            x = layer(x, illu_fea)

        return x

if __name__ == '__main__':
    # 测试EncoderResidualGroup
    x = torch.randn(2, 4, 128, 128)
    y = Illumination_Estimator(x)
    # encoder = EncoderResidualGroup(dim=64, num_heads=[1, 2, 4], num_blocks=3, ffn_expansion=4, LayerNorm_type='BiasFree', bias=True,n_fea_in=5)
    # y = encoder(x)
    # print(y.shape)