# -*- coding: utf-8 -*-
# File  : test.py
# Author: HeLei
# Date  : 2025/4/11
from torch import nn
from einops import rearrange
import torch
from torch.nn import functional as F


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle=64, n_fea_in=4):
        super().__init__()
        # 多尺度特征提取
        self.conv_low = nn.Sequential(
            nn.Conv2d(n_fea_in, n_fea_middle // 2, 3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv2d(n_fea_middle // 2, n_fea_middle // 2, 3, padding=2, dilation=2)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(n_fea_in, n_fea_middle // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_fea_middle // 2, n_fea_middle // 2, 1)
        )
        self.fusion = nn.Conv2d(n_fea_middle, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        mean_c = img.mean(dim=1, keepdim=True)
        input = torch.cat([img, mean_c], dim=1)  # [B,4,H,W]
        feat_low = self.conv_low(input)
        feat_high = self.conv_high(input)
        feat = torch.cat([feat_low, feat_high], dim=1)
        illu_map = self.sigmoid(self.fusion(feat))  # [B,1,H,W]
        return feat, illu_map

# class Illumination_Estimator(nn.Module):
#     """改进版光照估计器（输出单通道）"""
#
#     def __init__(self, n_fea_middle=32):
#         super().__init__()
#         self.conv1 = nn.Conv2d(4, n_fea_middle, 1)
#         self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, 5, padding=2, groups=n_fea_middle)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(n_fea_middle, 1, 1),
#             nn.Sigmoid()  # 约束光照图在[0,1]
#         )
#
#     def forward(self, img):
#         mean_c = img.mean(dim=1, keepdim=True)  # [B,1,H,W]
#         input = torch.cat([img, mean_c], dim=1)  # [B,4,H,W]
#         x = self.conv1(input)
#         x = self.depth_conv(x)
#         illu_map = self.conv2(x)
#         return x, illu_map  # 返回特征和光照图


class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            bias=False
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        # 使用卷积层生成 QKV
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        # 输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 位置编码
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea):
        b, c, h, w = x_in.shape  # [1, 3, 224, 224]

        # 如果输入通道数小于期望的 dim，则通过 1x1 卷积扩展通道数
        if c != self.dim:
            x_in = nn.Conv2d(c, self.dim, kernel_size=1, bias=False)(x_in)  # 扩展到指定维度

        # 生成 QKV
        qkv = self.qkv_dwconv(self.qkv(x_in))
        q, k, v = qkv.chunk(3, dim=1)  # [b, dim, h, w]

        # 将 illu_fea 调整为与 V 相同的通道数
        if illu_fea.shape[1] != v.shape[1]:  # 如果通道数不匹配，则进行调整
            illu_fea = nn.Conv2d(illu_fea.shape[1], v.shape[1], kernel_size=1, bias=False)(illu_fea)

        # 将 illu_fea 转换为多头形式
        illu_attn = rearrange(illu_fea, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 重排 QKV 以适应多头注意力机制
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 将 illu_attn 与 V 结合
        v1 = v * illu_attn  # 元素级乘法

        # 归一化 Q 和 K
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 加权求和得到输出
        out = (attn @ v1)

        # 恢复输出形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 输出投影
        out = self.project_out(out)

        # 添加位置编码
        pos_emb = self.pos_emb(out)
        out = out + pos_emb


        return out

if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    # Illumination Estimator
    illumination_estimator = Illumination_Estimator(n_fea_middle=48)
    illu_fea, illu_map = illumination_estimator(img)  # illu_fea: [1, 32, 224, 224], illu_map: [1, 3, 224, 224]
    print("illi_fea:", illu_fea.shape, "illi_map:", illu_map.shape)

    # Fusion Module
    # input_img = img * illu_map + img  # [1, 3, 224, 224]
    # fusion_module = IG_MSA(dim=64, dim_head=64, heads=8)
    # out = fusion_module(input_img, illu_fea)
    # print("out:", out.shape)


