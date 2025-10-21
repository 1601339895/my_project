# Copyright (c) 2022 Zhuang Intelligent Processing Lab. All rights reserved.
# Written by Zizheng Pan 

import math
import torch
import torch.nn as nn


class HiLo(nn.Module):
    """
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)  # 每个头的维度
        self.dim = dim  # 输入特征的维度

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)  # 低频注意力头数
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim  # 低频注意力特征维度

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads  # 高频注意力头数
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim  # 高频注意力特征维度

        # local window size. The `s` in our paper.
        self.ws = window_size  # 用于计算局部注意力的窗口大小

        if self.ws == 1:  # 如果窗口大小为1，则说明不需要局部注意力，则是全局注意力
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0  # 高频注意力头数为0
            self.h_dim = 0  # 高频注意力特征维度
            self.l_heads = num_heads  # 低频注意力头数为num_heads
            self.l_dim = dim  # 低频注意力特征维度

        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:  # 如果低频注意力头数大于0
            if self.ws != 1:  # 如果窗口大小不为1，则需要使用局部注意力
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)  # 局部注意力池化层
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)  # 低频注意力的Q
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)  # 低频注意力的K,V
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)  # 低频注意力的输出

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:  # 如果高频注意力头数大于0
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)  # 高频注意力的Q,K,V
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)  # 高频注意力的输出

    def hifi(self, x):
        B, C, H, W = x.shape  # B,C,H,W
        x = x.permute(0, 1, 2, 3)  # B,C,H,W -> B,H,W,C
        h_group, w_group = H // self.ws, W // self.ws  # 计算特征图在高度维度和宽度维度上的分组数

        total_groups = h_group * w_group  # 总的组数

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)   # [B, h_group, ws, w_group, ws,
        # C] -> [B, h_group*ws, w_group*ws, C]

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads,
                                    self.h_dim // self.h_heads).permute(3, 0, 1,4, 2, 5)  # 生成Q,K,V
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)  # B, H, W, C

        x = self.h_proj(x)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        return x

    def lofi(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B,H,W,C -> B,H,W,C

        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)  # [B,heads,HW,dim/heads]

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  #
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        return x

    def forward(self, x, ):

        if self.h_heads == 0:
            return self.lofi(x)

        if self.l_heads == 0:
            return self.hifi(x)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=1)

        return x

    def flops(self, H, W):
        # pad the feature map when the height and width cannot be divided by window size
        Hp = self.ws * math.ceil(H / self.ws)
        Wp = self.ws * math.ceil(W / self.ws)

        Np = Hp * Wp

        # For Hi-Fi
        # qkv
        hifi_flops = Np * self.dim * self.h_dim * 3
        nW = (Hp // self.ws) * (Wp // self.ws)
        window_len = self.ws * self.ws
        # q @ k and attn @ v
        window_flops = window_len * window_len * self.h_dim * 2
        hifi_flops += nW * window_flops
        # projection
        hifi_flops += Np * self.h_dim * self.h_dim

        # for Lo-Fi
        # q
        lofi_flops = Np * self.dim * self.l_dim
        kv_len = (Hp // self.ws) * (Wp // self.ws)
        # k, v
        lofi_flops += kv_len * self.dim * self.l_dim * 2
        # q @ k and attn @ v
        lofi_flops += Np * self.l_dim * kv_len * 2
        # projection
        lofi_flops += Np * self.l_dim * self.l_dim

        return hifi_flops + lofi_flops


if __name__ == '__main__':
    model = HiLo(dim=64, num_heads=8, window_size=2, alpha=0.5)
    x = torch.randn(1, 3, 224, 224)
    x1_Conv = nn.Conv2d(3, 64, 1, )
    x1 = x1_Conv(x)
    y = model(x1)
    print(y.shape)
    print(model.flops(224, 224))
