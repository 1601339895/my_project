# -*- coding: utf-8 -*-
# File  : model2.py
# Author: HeLei
# Date  : 2025/6/2
import math
from typing import List

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numbers
from timm.models.layers import DropPath


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class WithBias_Holistic_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(WithBias_Holistic_LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)  # 在通道和空间维度上进行归一化
        var = x.var(dim=(1, 2), unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return x_normalized * self.weight + self.bias


class BiasFree_Holistic_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(BiasFree_Holistic_LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return x_normalized * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='HolisticWithBias'):
        super(LayerNorm, self).__init__()
        self.dim = dim
        self.LayerNorm_type = LayerNorm_type

        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type == 'WithBias':
            self.body = WithBias_LayerNorm(dim)
        elif LayerNorm_type == 'HolisticWithBias':
            self.body = WithBias_Holistic_LayerNorm(dim)
        elif LayerNorm_type == 'HolisticBiasFree':
            self.body = BiasFree_Holistic_LayerNorm(dim)
        else:
            raise NotImplementedError(f"LayerNorm type {LayerNorm_type} is not implemented")

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Spatial_local_Enhanced(nn.Module):
    def __init__(self, dim, bias):
        super(Spatial_local_Enhanced, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        # 第一分支：普通卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1, dilation=1, groups=self.dim_sp, bias=bias),
            nn.BatchNorm2d(self.dim_sp),
            nn.GELU()
        )

        # 第二分支：空洞卷积（扩大感受野）
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=2, dilation=2, groups=self.dim_sp, bias=bias),
            nn.BatchNorm2d(self.dim_sp),
            nn.GELU()
        )

        # 融合后映射回原维度
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 空间注意力权重生成
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # 分通道处理
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        # 拼接 + 融合
        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.fuse(x_fused)

        # 空间注意力加权
        attn = self.spatial_gate(x_fused)
        out = x_fused * attn

        # 残差连接
        out = out + identity

        return out


class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x, 1, keepdim=True)[0]
        mean = torch.mean(x, 1, keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale = self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale


##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale


##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out


class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FreModule, self).__init__()

        # self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        # self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 8, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode='bilinear')

        high_feature, low_feature = self.fft(x)

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h // n * threshold[i, 0, :, :]).int()
            w_ = (w // n * threshold[i, 1, :, :]).int()

            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        fft = self.shift(fft)

        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low)

        return high, low


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class AdaptIR(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(AdaptIR, self).__init__()
        self.hidden = int(dim // ffn_expansion_factor)
        self.rank = self.hidden
        self.kernel_size = 3
        self.group = self.hidden
        self.head = nn.Conv2d(dim, self.hidden, 1, 1, bias=bias)

        self.BN = nn.BatchNorm2d(self.hidden)

        self.conv_weight_A = nn.Parameter(torch.randn(self.hidden, self.rank))
        self.conv_weight_B = nn.Parameter(
            torch.randn(self.rank, self.hidden // self.group * self.kernel_size * self.kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(self.hidden))
        nn.init.kaiming_uniform_(self.conv_weight_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv_weight_B, a=math.sqrt(5))

        self.amp_fuse = nn.Conv2d(self.hidden, self.hidden, 1, 1, groups=self.hidden, bias=bias)
        self.pha_fuse = nn.Conv2d(self.hidden, self.hidden, 1, 1, groups=self.hidden, bias=bias)
        nn.init.ones_(self.pha_fuse.weight)
        nn.init.ones_(self.amp_fuse.weight)
        nn.init.zeros_(self.amp_fuse.bias)
        nn.init.zeros_(self.pha_fuse.bias)

        self.compress = nn.Conv2d(self.hidden, 1, 1, 1, bias=bias)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2, bias=bias),
            nn.GELU(),
            nn.Linear(self.hidden // 2, self.hidden, bias=bias),
        )

        self.tail = nn.Conv2d(self.hidden, dim, 1, 1, bias=bias)
        nn.init.zeros_(self.tail.weight)

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.hidden, self.hidden // 4, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(self.hidden // 4, self.hidden, kernel_size=1, bias=bias)
        )
        nn.init.zeros_(self.channel_interaction[3].weight)
        nn.init.zeros_(self.channel_interaction[3].bias)

        self.spatial_interaction = nn.Conv2d(self.hidden, 1, kernel_size=1, bias=bias)
        nn.init.zeros_(self.spatial_interaction.weight)
        nn.init.zeros_(self.spatial_interaction.bias)

    def forward(self, x):
        # N, H, W, C = x.shape
        N, C, H, W = x.shape
        # x = x.permute(0, 3, 1, 2).contiguous()  # N,C,H,W
        x = self.BN(self.head(x))

        global_x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        mag_x = torch.abs(global_x)
        pha_x = torch.angle(global_x)
        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        real = Mag * torch.cos(Pha)
        imag = Mag * torch.sin(Pha)
        global_x = torch.complex(real, imag)
        global_x = torch.fft.irfft2(global_x, s=(H, W), dim=(2, 3), norm='ortho')  # N,C,H,W
        global_x = torch.abs(global_x)

        conv_weight = (self.conv_weight_A @ self.conv_weight_B) \
            .view(self.hidden, self.hidden // self.group, self.kernel_size, self.kernel_size).contiguous()
        local_x = F.conv2d(x, weight=conv_weight, bias=self.conv_bias, stride=1, padding=1, groups=self.group)

        score = self.compress(x).view(N, 1, H * W).permute(0, 2, 1).contiguous()  # N,HW,1
        score = F.softmax(score, dim=1)
        out = x.view(N, self.hidden, H * W)  # N,C,HW
        out = out @ score  # N,C,1
        out = out.permute(2, 0, 1)  # 1,N,C
        out = self.proj(out)
        channel_score = out.permute(1, 2, 0).unsqueeze(-1).contiguous()  # N,C,1,1

        channel_gate = self.channel_interaction(global_x).sigmoid()
        spatial_gate = self.spatial_interaction(local_x).sigmoid()
        spatial_x = channel_gate * local_x + spatial_gate * global_x

        x = self.tail(channel_score * spatial_x)
        # x = x.permute(0, 2, 3, 1).contiguous()
        return x


class Mix(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type, bias):
        super(Mix, self).__init__()
        self.dim = dim

        self.norms = nn.ModuleList([
            LayerNorm(dim // 2, LayerNorm_type),
            LayerNorm(dim // 2, LayerNorm_type)
        ])
        self.Gloal = FreModule(dim // 2, num_heads, bias)
        self.Local = Spatial_local_Enhanced(dim // 2, bias)

        # channel attention
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()
        self.init_conv = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
        )
        self.finall_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x, y):
        # x: (B, C, H, W)
        x_attn = x[:, 1::2, :, :]
        x_local = x[:, 0::2, :, :]

        x_attn = self.norms[0](x_attn)
        x_attn = self.Gloal(y, x_attn)
        x_local = self.norms[1](x_local)
        x_local = self.Local(x_local)
        fuse = torch.cat([x_attn, x_local], dim=1)
        x = self.gelu(fuse)
        out = self.ca(x) * x
        return out


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, LayerNorm_type, bias):
        super().__init__()
        drop_path = 0.0
        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),  #
            LayerNorm(dim, LayerNorm_type)  #
        ])

        self.mixer = Mix(dim, num_heads, LayerNorm_type, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adaptir = AdaptIR(dim, ffn_expansion_factor, bias=bias)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x, y):
        short_cut = x
        x = self.norms[0](x)
        x = self.mixer(x, y) * self.beta
        x = short_cut + x

        short_cut = x
        x = self.norms[1](x)
        adapt = self.adaptir(x)
        x = self.ffn(x)
        x = short_cut + self.drop_path(x + adapt) * self.gamma
        # x = x + self.mixer(self.norms[0](x)) * self.beta
        # x = x + self.ffn(self.norms[1](x)) * self.gamma
        return x


class EncoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion_factor: int,
                 LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                EncoderBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                             LayerNorm_type=LayerNorm_type, bias=bias, )
            )

    def forward(self, x, y):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x, y)
            i += 1
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, LayerNorm_type, bias):
        super().__init__()
        drop_path = 0.0
        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),  #
            LayerNorm(dim, LayerNorm_type)  #
        ])

        self.mixer = Mix(dim, num_heads, LayerNorm_type, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.mlp = Mlp(dim, bias=bias)
        self.adaptir = AdaptIR(dim, ffn_expansion_factor, bias=bias)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x, y):
        short_cut = x
        x = self.norms[0](x)
        x = self.mixer(x, y) * self.beta
        x = short_cut + x

        short_cut = x
        x = self.norms[1](x)
        adapt = self.adaptir(x)
        x = self.ffn(x)
        x = short_cut + self.drop_path(x + adapt) * self.gamma
        return x


class DecoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion_factor: int,
                 LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                DecoderBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                             LayerNorm_type=LayerNorm_type, bias=bias, )
            )

    def forward(self, x, y):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x, y)
            i += 1
        return x


class MyModel(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 levels=4,
                 num_blocks=[4, 6, 6, 8],
                 num_dec_blocks=[1, 1, 1],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 decoder=True,
                 ):
        super(MyModel, self).__init__()
        self.leves = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks

        dims = [dim * 2 ** i for i in range(levels)]

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)

        # -- Encoder --
        self.enc = nn.ModuleList([])
        for i in range(levels - 1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i],
                    num_blocks=num_blocks[i],
                    num_heads=heads[i],
                    ffn_expansion_factor=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type,
                    bias=True, ),
                Downsample(dim * 2 ** i)
            ])
            )
        # -- Latent --
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1],
            num_heads=heads[-1],
            ffn_expansion_factor=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True, )
        # -- Decoder --
        dims = dims[::-1]
        heads = heads[::-1]
        num_dec_blocks = num_dec_blocks[::-1]

        self.dec = nn.ModuleList([])
        for i in range(levels - 1):
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, bias=bias),
                DecoderResidualGroup(
                    dim=dims[i],
                    num_blocks=num_dec_blocks[i],
                    num_heads=heads[i],
                    ffn_expansion_factor=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type,
                    bias=True, ),
            ])
            )
        # -- Refinement --
        heads = heads[::-1]
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks,
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True, )
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.total_loss = None

    def forward(self, x, labels=None):

        feats = self.patch_embed(x)

        self.total_loss = 0
        enc_feats = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats, x)
            enc_feats.append(feats)
            feats = downsample(feats)

        feats = self.latent(feats, x)

        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            feats = block(feats, x)
            self.total_loss += block.loss

        feats = self.refinement(feats, x)
        x = self.output(feats) + x

        self.total_loss /= sum(self.num_dec_blocks)
        return x


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    model = MyModel()
    out = model(img)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {model_params / 1e6:.3f} M")
    print(out.shape)
