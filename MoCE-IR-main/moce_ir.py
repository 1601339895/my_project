from collections import OrderedDict
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import numbers
import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal
from fvcore.nn import FlopCountAnalysis, flop_count_table


##########################################################################
## Helper functions
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class MySequential(nn.Sequential):
    def forward(self, x1, x2):
        # Iterate through all layers in sequential order
        for layer in self:
            # Check if the layer takes two inputs (i.e., custom layers)
            if isinstance(layer, nn.Module):
                # Pass both inputs to the layer
                x1 = layer(x1, x2)
            else:
                # For non-module layers, pass the two inputs directly
                x1 = layer(x1, x2)
        return x1

def softmax_with_temperature(logits, temperature=1.0):
    """
    Apply softmax with temperature to the logits.
    
    Args:
    - logits (torch.Tensor): The input logits.
    - temperature (float): The temperature factor.
    
    Returns:
    - torch.Tensor: The softmax output with temperature.
    """
    # Scale the logits by the temperature
    scaled_logits = logits / temperature
    
    # Apply softmax
    return F.softmax(scaled_logits, dim=-1)

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined
    
    def to_spatial(self, x, x_shape):
        h, w = x_shape
        amp, phase = x.chunk(2, dim=1)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        x = real + 1j * imag
        x = torch.fft.ifft2(x, s=(h, w), norm="backward").real
        return x

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


##########################################################################
class GELU(nn.Module):
    """
    激活函数 GELU
    """

    def forward(self, x):
        return F.gelu(x)

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class HighPassConv2d(nn.Module):
    def __init__(self, c, freeze):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=c
        )

        kernel = torch.tensor([[[[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]]]], dtype=torch.float32)
        self.conv.weight.data = kernel.repeat(c, 1, 1, 1)

        if freeze:
            self.conv.requires_grad_ = False

    def forward(self, x):
        return self.conv(x)


class DynamicFusionGate(nn.Module):
    """动态光照融合门控模块"""

    def __init__(self, in_channels=3, gate_channels=16):
        super().__init__()
        # 轻量级门控网络
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, gate_channels, 3, padding=1),  # 输入: img + illu_map (都是3通道)
            nn.GELU(),
            nn.Conv2d(gate_channels, gate_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(gate_channels // 2, 2, 1),  # 输出两个分支的权重
            nn.Softmax(dim=1)  # 保证权重和为1
        )

    def forward(self, img, illu_map):
        """
        Args:
            img: 原始输入图像 [B,3,H,W]
            illu_map: 光照图 [B,3,H,W]
        Returns:
            fused_img: 融合后的图像 [B,3,H,W]
        """
        # 拼接输入图像和光照图 (都是3通道)
        cat_input = torch.cat([img, illu_map], dim=1)  # [B,6,H,W]

        # 计算门控权重
        gate_weights = self.conv(cat_input)  # [B,2,H,W]

        # 动态加权融合
        # 对于每个通道分别计算融合结果
        fused_img = (gate_weights[:, 0:1] * img / (illu_map + 1e-6) +
                     gate_weights[:, 1:2] * (img + illu_map * img))
        return fused_img


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
#     """
#     用于估计光照信息的网络结构
#     """
#
#     def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
#         """
#         :param n_fea_middle: 中间特征图的通道数
#         :param n_fea_in: 输入特征图的通道数，默认值为 4
#         :param n_fea_out: 输出特征图的通道数，默认值为 3
#         """
#         super(Illumination_Estimator, self).__init__()
#
#         self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1,
#                                bias=True)  # 将输入特征图的通道数从n_fea_in变为n_fea_middle，这里使用了1x1的卷积核，即没有空间上的信息变化，只进行通道数的变化。
#
#         self.depth_conv = nn.Conv2d(
#             n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True,
#             groups=n_fea_middle)  # 在通道维度上进行卷积，深度可分离卷积。
#         self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1,
#                                bias=True)  # 将通道数从n_fea_middle变为n_fea_out，这里使用了1x1的卷积核
#
#     def forward(self, img):
#         # img:        b,c=3,h,w
#         # mean_c:     b c=1,h,w
#
#         # illu_fea:   b,c,h,w
#         # illu_map:   b,c=3,h,w
#
#         mean_c = img.mean(dim=1).unsqueeze(1)  # 计算输入图像在每个通道上的均值，并在通道维度上进行扩展，形状为 [b,1,h,w]
#         input = torch.cat([img, mean_c], dim=1)  # 将原始输入图像和均值特征拼接在一起，形状为 [b,c+1,h,w]
#
#         x_1 = self.conv1(input)  # 依次通过卷积进行特征提取
#         illu_fea = self.depth_conv(x_1)
#         illu_map = self.conv2(illu_fea)
#         return illu_fea, illu_map  # 返回光照特征图，形状为[batch_size, n_fea_middle, h, w]和光照估计图，形状为[batch_size, n_fea_out, h, w]


class AdaptiveIlluminationModulator(nn.Module):
    def __init__(self, dim):
        super(AdaptiveIlluminationModulator, self).__init__()
        # 使用卷积层来学习光照特征的调制参数
        self.modulate_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()  # 输出范围为 [0, 1]，用于调节强度
        )

    def forward(self, illu_fea):
        """
        Args:
            illu_fea: 光照特征 [B, C, H, W]
        Returns:
            modulator: 调制参数 [B, C, H, W]
        """
        modulator = self.modulate_conv(illu_fea)
        return modulator


class AttentionWithIllumination(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(AttentionWithIllumination, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # QKV 分支
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)

        # 输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 空间位置编码（pos_emb）
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim)
        )

        # 自适应光照调制器
        self.illumination_modulator = AdaptiveIlluminationModulator(dim)

    def forward(self, x, illu_fea):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
            illu_fea: 光照特征 [B, C, H, W]，用于调制注意力
        Returns:
            out: 输出特征图 [B, C, H, W]
        """
        b, c, h, w = x.shape

        # ========== 自适应光照调制 ==========
        illu_mod = self.illumination_modulator(illu_fea)  # 根据光照特征生成调制参数

        # ========== 计算 QKV ==========
        qkv = self.qkv_dwconv(self.qkv(x))
        q_inp, k_inp, v_inp = qkv.chunk(3, dim=1)

        # ========== 注意力计算 ==========        q = rearrange(q_inp, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k_inp, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_inp, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 对 v 应用光照调制
        illu_attn = rearrange(illu_mod, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = v * illu_attn

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # ========== 输出投影 ==========
        out_c = self.project_out(out)

        # ========== 对 v_inp 进行位置编码 ==========
        # v_inp = rearrange(v_inp, 'b (head c) h w -> b c h w', head=self.num_heads)
        pos_emb = self.pos_emb(v_inp)

        # ========== 最终输出 + 残差连接 + pos_emb ==========
        out = out_c + pos_emb  # 在最后加上位置编码作为残差

        return out


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

class New_EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type)
        ])

        self.mixer = AttentionWithIllumination(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x, illu_fea):
        x = self.norms[0](x)
        x = x + self.mixer(x, illu_fea)
        x = self.norms[1](x)
        x = x + self.ffn(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type)
        ])

        self.mixer = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.mixer(self.norms[0](x))
        x = x + self.ffn(self.norms[1](x))
        return x


class EncoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type)
            )

    def forward(self, x):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x)
            i += 1
        return x


    ##########################################################################
## Multi-DConv Head Transposed Self-Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, stride=1, padding=7//2, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, y):
        b,c,h,w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
##########################################################################
## Self-Attention in Fourier Domain
class FFTAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super(FFTAttention, self).__init__()

        self.patch_size = kwargs["patch_size"]
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, stride=1, padding=7//2, groups=dim*2)
        self.norm = LayerNorm(dim, "WithBias")
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)        
        
    def pad_and_rearrange(self, x):
        b, c, h, w = x.shape
        
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        x = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        return x
    
    def rearrange_to_original(self, x, x_shape):
        h, w = x_shape
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        x = x[:, :, :h, :w]  # Slice out the original height and width
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
            
        q = self.pad_and_rearrange(q)
        k = self.pad_and_rearrange(k)
            
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        
        out = self.rearrange_to_original(out, (h, w))
        
        out = self.norm(out)
        out = out * v
        
        out = self.proj_out(out)
        return out

    
     
##########################################################################
## Adapter Block    
class ModExpert(nn.Module):
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size:int):
        super(ModExpert, self).__init__()
        
        self.depth = depth
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)
            
    def process(self, x, shared):
        shortcut = x
        x = self.proj[0](x)
        x = self.body(x) * F.silu(self.proj[1](shared))
        x = self.proj[2](x)
        return x + shortcut

    def feat_extract(self, feats, shared):
        for _ in range(self.depth):
            feat = self.process(feats, shared)
        return feat
    
    def forward(self, x, shared):
        b, c, h, w = x.shape
        
        if b == 0:
            return x
        else:
            x = self.feat_extract(x, shared)
            return x
        



########################################################################### 
## Adapter Layer
class AdapterLayer(nn.Module):
    def __init__(self, 
                 dim: int, rank: int, num_experts: int = 4, top_k: int=2, expert_layer: nn.Module=FFTAttention, stage_depth: int=1,
                 depth_type: str="lin", rank_type: str="constant", freq_dim: int=128, 
                 with_complexity: bool=False, complexity_scale: str="min"):
        super().__init__()            
        
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.noise_eps = 1e-2
        self.num_experts = num_experts

        patch_sizes = [2**(i+2) for i in range(num_experts)]
        kernel_sizes = [3+(2*i) for i in range(num_experts)]
        
        if depth_type == "lin":
            depths = [stage_depth+i for i in range(num_experts)]
        elif depth_type == "double":
            depths = [stage_depth+(2*i) for i in range(num_experts)]
        elif depth_type == "exp":
            depths = [2**(i) for i in range(num_experts)]
        elif depth_type == "fact":
            depths = [math.factorial(i+1) for i in range(num_experts)]
        elif isinstance(depth_type, int):
            depths = [depth_type for _ in range(num_experts)]
        elif depth_type == "constant":
            depths = [stage_depth for i in range(num_experts)]
        else:
            raise(NotImplementedError)
        
        if rank_type == "constant":
            ranks = [rank for _ in range(num_experts)]
        elif rank_type == "lin":
            ranks = [rank+i for i in range(num_experts)]
        elif rank_type == "double":
            ranks = [rank+(2*i) for i in range(num_experts)]
        elif rank_type == "exp":
            ranks = [rank**(i+1) for i in range(num_experts)]
        elif rank_type == "fact":
            ranks = [math.factorial(rank+i) for i in range(num_experts)]
        elif rank_type == "spread":
            ranks = [dim//(2**i) for i in range(num_experts)][::-1]
        else:
            raise(NotImplementedError)
        
        self.experts = nn.ModuleList([
            MySequential(*[ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel)])
            for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
        ])
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, freq_dim, 
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale
        )
        
    def forward(self, x, freq_emb, shared):
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x, freq_emb)
        self.loss = aux_loss
                
        # routing
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_intputs = dispatcher.dispatch(shared)
            expert_outputs = [self.experts[exp](expert_inputs[exp], expert_shared_intputs[exp]) for exp in range(len(self.experts))]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            selected_experts = [self.experts[i] for i in top_k_indices.squeeze(0)]  # Select the corresponding experts
            expert_outputs = torch.stack([expert(x, shared) for expert in selected_experts], dim=1)
            gates = gates.gather(1, top_k_indices)  
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs 
            out = weighted_outputs.sum(dim=1)  # Sum across the top-k dimension to get the final output
            
        out = self.proj_out(out)
        return out

    

class RoutingFunction(nn.Module):
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias: bool = True, complexity_scale: str="max"):
        super(RoutingFunction, self).__init__()
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        ) 
        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)
        
        self.k = k
        self.tau = 1
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias

    def forward(self, x, freq_emb):
        logits = self.gate(x) + self.freq_gate(freq_emb)
        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))
        
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        # Final auxiliary loss
        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0
        
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values, aux_loss

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)
        importance = importance * (self.complexity * self.tau) if self.use_complexity_bias else importance
        imp_mean = importance.mean()
        imp_std = importance.std()
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2
        return loss_imp

    def load_loss(self, logits, logits_noisy, noise_std):
        # Compute the noise threshold
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        
        # Compute the load for each expert
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )
        
        # Calculate noise required to win
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        
        # Probability of being above the threshold
        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)
        
        # Compute mean probability for each expert over examples
        p_mean = p.mean(dim=0)
        
        # Compute p_mean's coefficient of variation squared
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2
        
        return loss_load


##########################################################################
## Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, expert_layer, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, depth_type=None, rank_type=None, stage_depth=None, freq_dim:int=128, with_complexity: bool=False):
        super().__init__()

        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type),
        ])
        
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        ])
        
        self.shared = Attention(dim, num_heads, bias)
        self.mixer = CrossAttention(dim, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        
        self.adapter = AdapterLayer(
            dim, rank, 
            top_k=top_k, num_experts=num_experts, expert_layer=expert_layer, freq_dim=freq_dim,
            depth_type=depth_type, rank_type=rank_type, stage_depth=stage_depth, 
            with_complexity=with_complexity, complexity_scale=complexity_scale
        )
        
    def forward(self, x, freq_emb=None):    
        shortcut = x
        x = self.norms[0](x)
        
        x_s = self.proj[0](x)
        x_a = self.proj[1](x)
        x_s = self.shared(x_s)
        x_a = self.adapter(x_a, freq_emb, x_s)
        x = self.mixer(x_a, x_s) + shortcut

        x = x + self.ffn(self.norms[1](x))
        return x, self.adapter.loss



######################################################################
## Encoder Residual Group
class New_EncoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                New_EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type)
            )

    def forward(self, x,illu_fea):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x,illu_fea)
            i += 1
        return x


    
######################################################################
## Decoder Residual Group
class DecoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool, complexity_scale=None,
                 rank=None, num_experts=None, expert_layer=None, top_k=None, depth_type=None, stage_depth=None, rank_type=None, freq_dim:int=128, with_complexity: bool=False):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                DecoderBlock(
                    dim, num_heads, ffn_expansion, bias, LayerNorm_type, 
                    expert_layer=expert_layer, rank=rank, num_experts=num_experts, top_k=top_k, 
                    stage_depth=stage_depth, freq_dim=freq_dim, complexity_scale=complexity_scale,
                    depth_type=depth_type, rank_type=rank_type, with_complexity=with_complexity
                )
            )

    def forward(self, x, freq_emb=None):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x , loss = self.layers[i](x, freq_emb)
            self.loss += loss
            i += 1
        return x  
    
    
     
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.estimator = Illumination_Estimator(n_fea_middle=embed_dim)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        _,illu_map = self.estimator(x)
        x = x * illu_map + x
        x = self.proj(x)
        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
    
    
##########################################################################
## Frequency Embedding
class FrequencyEmbedding(nn.Module):
    """
    Embeds magnitude and phase features extracted from the bottleneck of the U-Net.
    """
    def __init__(self, dim):
        super(FrequencyEmbedding, self).__init__()
        self.high_conv = nn.Sequential(
            HighPassConv2d(dim, freeze=True),
            nn.GELU())
        
        self.mlp= nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, dim)
            )

    def forward(self, x):
        x = self.high_conv(x)
        x = x.mean(dim=(-2, -1))
        x = self.mlp(x)
        return x
    
    
##########################################################################
##
class MoCEIR(nn.Module):
    def __init__(self,
                inp_channels=3, 
                out_channels=3, 
                dim = 32,
                levels: int = 4,
                heads = [1,1,1,1],
                num_blocks = [1,1,1,3],
                num_dec_blocks = [1, 1, 1],
                ffn_expansion_factor = 2.66,
                num_refinement_blocks = 1,
                LayerNorm_type = 'WithBias', ## Other option 'BiasFree'
                bias = False,
                rank=2,
                num_experts=4,
                depth_type="lin",
                stage_depth=[3,2,1],
                rank_type="constant",
                topk=1,
                expert_layer=FFTAttention,
                with_complexity=False,
                complexity_scale="max",
                ):
        super(MoCEIR, self).__init__()

        self.dim = dim
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.padder_size = 2 ** len(num_blocks)
        
        dims = [dim*2**i for i in range(levels)]
        ranks = [rank for i in range(levels-1)]

        self.estimator = Illumination_Estimator(n_fea_middle=dim)  # 光照信息估计

        self.fuse = DynamicFusionGate()

        # -- Patch Embedding
        # self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        self.embedding = nn.Conv2d(inp_channels, self.dim, 3, 1, 1, bias=False)
        self.freq_embed = FrequencyEmbedding(dims[-1])
                
        # -- Encoder --        
        self.enc = nn.ModuleList([])
        for i in range(levels-1):
            self.enc.append(nn.ModuleList([
                New_EncoderResidualGroup(
                    dim=dims[i], 
                    num_blocks=num_blocks[i], 
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=True,),
                Downsample(dim*2**i)
                ])
            )
        
        # -- Latent --
        self.latent = New_EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1], 
            num_heads=heads[-1], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
                  
        # -- Decoder --
        dims = dims[::-1]
        ranks = ranks[::-1]
        heads = heads[::-1]
        num_dec_blocks = num_dec_blocks[::-1]
        
        self.dec = nn.ModuleList([])
        for i in range(levels-1):
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=1, bias=bias),
                DecoderResidualGroup(
                    dim=dims[i+1],
                    num_blocks=num_dec_blocks[i], 
                    num_heads=heads[i+1],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=bias, expert_layer=expert_layer, freq_dim=dims[0], with_complexity=with_complexity,
                    rank=ranks[i], num_experts=num_experts, stage_depth=stage_depth[i], depth_type=depth_type, rank_type=rank_type, top_k=topk, complexity_scale=complexity_scale),
                ])
            )

        # -- Refinement --
        heads = heads[::-1]
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks, 
            num_heads=heads[0], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
        
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.total_loss = None
    
    def forward(self, img, labels=None):

        x = self.check_image_size(img)


        illu_fea, illu_map = self.estimator(img)

        x = self.fuse(x, illu_map)

        # x = x * illu_map + x

        feats = self.embedding(x)

        # feats = self.patch_embed(x)
        
        self.total_loss = 0
        enc_feats = []
        # illu_fea_list = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats,illu_fea)
            enc_feats.append(feats)
            # illu_fea_list.append(illu_fea)
            feats = downsample(feats)
            illu_fea = downsample(illu_fea)
        
        feats = self.latent(feats,illu_fea)
        freq_emb = self.freq_embed(feats)
                
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            feats = block(feats, freq_emb)
            self.total_loss += block.loss

        feats = self.refinement(feats)
        out = self.output(feats) + img

        self.total_loss /= sum(self.num_dec_blocks)
        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # 需要填充的高度和宽度，以确保图像的高度和宽度是 self.padder_size 的倍数。
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)  # 使用F.pad函数填充图像。
        return x
                    
    
    
if __name__ == "__main__":
    # test
    model = MoCEIR(rank=2, num_blocks=[4,6,6,8],
                   num_dec_blocks=[2,4,4],
                   levels=4,
                   dim=32,
                   num_refinement_blocks=4,
                   with_complexity=True,
                   complexity_scale="max",
                   stage_depth=[1,1,1],
                   depth_type="constant",
                   rank_type="spread",
                   num_experts=4,
                   topk=1,
                   expert_layer=FFTAttention).cuda()
    # 计算模型参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {model_params / 1e6:.3f} M")

    x = torch.randn(1, 3, 224, 224).cuda()
    _ = model(x)
    print(model.total_loss)
    # Memory usage
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))

    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))
