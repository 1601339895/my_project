# -*- coding: utf-8 -*-
# File  : igmoce_ir.py.py
# Author: HeLei
# Date  : 2025/5/13
import numbers
from typing import List

import torch
import math
import numbers
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal
from fvcore.nn import FlopCountAnalysis, flop_count_table


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

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)
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


class FreMLP(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(dim, dim * ffn_expansion_factor, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim * ffn_expansion_factor, dim, kernel_size=1, bias=bias))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Denoise_1(nn.Module):
    def __init__(self, fea_middle=48, in_channels=3, out_channels=3):
        super(Denoise_1, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, fea_middle, 3, padding=1)
        self.conv2 = nn.Conv2d(fea_middle, fea_middle, 3, padding=1)
        self.conv3 = nn.Conv2d(fea_middle, out_channels, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Illu_Estimator(nn.Module):
    def __init__(self, layers, dim):
        super(Illu_Estimator, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # self.proj = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input):
        illu_fea = self.in_conv(input)
        for conv in self.blocks:
            illu_fea = illu_fea + conv(illu_fea)
        illu_fea = self.out_conv(illu_fea)
        illu_fea = torch.clamp(illu_fea, 0.0001, 1)

        # illu_fea = self.proj(illu_fea)

        return illu_fea  # [b,3,h,w]


class IG_MSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(IG_MSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)

        self.proj = nn.Linear(dim, dim, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x_in, illu_fea):

        b, c, h, w = x_in.shape
        x = x_in.reshape(b, c, h * w).permute(0, 2, 1)

        q_inp = self.to_q(x)  #
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q = rearrange(q_inp, 'b n (h d) -> b h n d', h=self.num_heads)  # [b,h,n,d]
        k = rearrange(k_inp, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v_inp, 'b n (h d) -> b h n d', h=self.num_heads)


        illu_attn = illu_fea
        illu_attn = rearrange(illu_attn, 'b d h w -> b d (h w)')  # [b, d, n]
        illu_attn = rearrange(illu_attn, 'b (h d) n -> b h n d', h=self.num_heads)  # [b, h, n, d]


        v = v * illu_attn

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1,p=2)
        k = F.normalize(k, dim=-1,p=2)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)


        out = (attn @ v)  # [b,h,d,hw]
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.permute(0,3,1,2)  # [b,hw,h,d]
        out = out.reshape(b,h*w,self.dim)  # [b,hw,d]

        out_c = self.proj(out).view(b,c,h,w)

        pos_emb = self.pos_emb(v_inp.reshape(b, c, h, w))

        out = out_c + pos_emb

        # out = out + self.gamma * x_in

        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, stride=1, padding=7 // 2, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

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

class SpAM(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, dim, DW_Expand=2, dilations=[1], extra_depth_wise=False, ):
        super().__init__()

        self.dw_channel = DW_Expand * dim
        self.extra_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(dim, DW_Expand, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * dim
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, inp):
        shortcut = inp
        x = self.conv1(self.extra_conv(inp))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = shortcut + self.beta * x

        return y


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


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, DW_Expand=2, dilations=[1, 2, 4],
                 extra_depth_wise=False, fusion_weight=0.5):
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim//2, LayerNorm_type),
            LayerNorm(dim//2, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type),
        ])

        self.mixer = IG_MSA(dim//2, num_heads, bias)
        self.spam = SpAM(dim//2, DW_Expand=DW_Expand, dilations=dilations, extra_depth_wise=extra_depth_wise)
        self.ffn = FreMLP(dim, ffn_expansion_factor, bias)

        self.fusion_weight = fusion_weight  # 融合权重

        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.embedding = nn.Conv2d(3, dim, 3, 1, 1, bias=False)

    def spatial_fusion(self, F_att, F_gate):
        F_gq = F_att + torch.sigmoid(F_gate)
        F_ga = F_gate + torch.sigmoid(F_att)
        combined_features = torch.cat((F_gq, F_ga), dim=1)
        return combined_features

    def frequency_fusion(self, F_att, F_gate):
        # 进行傅里叶变换
        F_att_fft = torch.fft.rfft2(F_att)
        F_gate_fft = torch.fft.rfft2(F_gate)
        # 频域融合
        F_fft = F_att_fft + F_gate_fft
        # 逆变换回时域
        F_freq = torch.fft.irfft2(F_fft, s=F_att.shape[-2:])
        F_freq = F_freq.repeat(1, F_att.shape[1] * 2 // F_freq.shape[1], 1, 1)

        return F_freq

    def forward(self, x, illu_fea):
        b, c, h, w = x.shape
        shortcut = x
        x = self.norms[0](x)  # [b,c,h,w]

        # 进行特征分割
        attn = x[:, 1::2, :, :]
        gate = x[:, 0::2, :, :]
        # illu_fea = illu_fea[:, 1::2, :, :]

        # 进行IG-MSA和SpAM
        attn = self.norms[1](attn)
        gate = self.norms[2](gate)

        F_att = self.mixer(attn, illu_fea)
        F_gate = self.spam(gate)

        # 空间融合
        F_s = self.spatial_fusion(F_att, F_gate)
        # 频率融合
        F_f = self.frequency_fusion(F_att, F_gate)
        # 加权融合
        F_fuse = self.fusion_weight * F_s + (1 - self.fusion_weight) * F_f  # [b,c,h,w]

        F_fuse = F_fuse * self.gamma + shortcut

        # Second Step
        F_fuse = self.norms[3](F_fuse)
        # F_fuse_flat = F_fuse.view(F_fuse.size(0), -1)
        output = self.ffn(F_fuse)

        output = output.view(b, c, h, w)
        output = F_fuse + self.beta * output

        return output


class EncoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, DW_Expand: int,
                 dilations: List[int], fusion_weight: float, LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                # EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type)
                EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type, DW_Expand=DW_Expand,
                             dilations=dilations, fusion_weight=fusion_weight)
            )
    def forward(self, x,illu_fea):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x,illu_fea)
            i += 1
        return x


## Adapter Block
class ModExpert(nn.Module):
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size: int):
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


## Self-Attention in Fourier Domain
class FFTAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super(FFTAttention, self).__init__()

        self.patch_size = kwargs["patch_size"]

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, stride=1, padding=7 // 2, groups=dim * 2)
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

###########################################################################
## Adapter Layer
class AdapterLayer(nn.Module):
    def __init__(self,
                 dim: int, rank: int, num_experts: int = 4, top_k: int = 2, expert_layer: nn.Module = FFTAttention,
                 stage_depth: int = 1,
                 depth_type: str = "lin", rank_type: str = "constant", freq_dim: int = 128,
                 with_complexity: bool = False, complexity_scale: str = "min"):
        super().__init__()

        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.noise_eps = 1e-2
        self.num_experts = num_experts

        patch_sizes = [2 ** (i + 2) for i in range(num_experts)]
        kernel_sizes = [3 + (2 * i) for i in range(num_experts)]

        if depth_type == "lin":
            depths = [stage_depth + i for i in range(num_experts)]
        elif depth_type == "double":
            depths = [stage_depth + (2 * i) for i in range(num_experts)]
        elif depth_type == "exp":
            depths = [2 ** (i) for i in range(num_experts)]
        elif depth_type == "fact":
            depths = [math.factorial(i + 1) for i in range(num_experts)]
        elif isinstance(depth_type, int):
            depths = [depth_type for _ in range(num_experts)]
        elif depth_type == "constant":
            depths = [stage_depth for i in range(num_experts)]
        else:
            raise (NotImplementedError)

        if rank_type == "constant":
            ranks = [rank for _ in range(num_experts)]
        elif rank_type == "lin":
            ranks = [rank + i for i in range(num_experts)]
        elif rank_type == "double":
            ranks = [rank + (2 * i) for i in range(num_experts)]
        elif rank_type == "exp":
            ranks = [rank ** (i + 1) for i in range(num_experts)]
        elif rank_type == "fact":
            ranks = [math.factorial(rank + i) for i in range(num_experts)]
        elif rank_type == "spread":
            ranks = [dim // (2 ** i) for i in range(num_experts)][::-1]
        else:
            raise (NotImplementedError)

        self.experts = nn.ModuleList([
            MySequential(
                *[ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel)])
            for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
        ])

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, freq_dim,
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale
        )

    def forward(self, x, shared):
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x)
        self.loss = aux_loss

        # routing
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_intputs = dispatcher.dispatch(shared)
            expert_outputs = [self.experts[exp](expert_inputs[exp], expert_shared_intputs[exp]) for exp in
                              range(len(self.experts))]
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
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias: bool = True,
                 complexity_scale: str = "max"):
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

    def forward(self, x):
        logits = self.gate(x)
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


## Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, expert_layer, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, depth_type=None, rank_type=None, stage_depth=None,
                 freq_dim: int = 128, with_complexity: bool = False,DW_Expand=2, dilations=[1, 2, 4],extra_depth_wise=False):
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type),
        ])

        self.proj = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        ])

        self.shared = SpAM(dim, DW_Expand=DW_Expand, dilations=dilations, extra_depth_wise=extra_depth_wise)
        self.mixer = CrossAttention(dim, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.adapter = AdapterLayer(
            dim, rank,
            top_k=top_k, num_experts=num_experts, expert_layer=expert_layer, freq_dim=freq_dim,
            depth_type=depth_type, rank_type=rank_type, stage_depth=stage_depth,
            with_complexity=with_complexity, complexity_scale=complexity_scale
        )

    def forward(self, x):
        shortcut = x
        x = self.norms[0](x)

        x_s = self.proj[0](x)
        x_a = self.proj[1](x)
        x_s = self.shared(x_s)
        x_a = self.adapter(x_a, x_s)
        x = self.mixer(x_a, x_s) + shortcut

        x = x + self.ffn(self.norms[1](x))
        return x, self.adapter.loss

## Decoder Residual Group
class DecoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool,
                 complexity_scale=None,
                 rank=None, num_experts=None, expert_layer=None, top_k=None, depth_type=None, stage_depth=None,
                 rank_type=None, freq_dim: int = 128, with_complexity: bool = False):
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

    def forward(self, x,):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x, loss = self.layers[i](x)
            self.loss += loss
            i += 1
        return x

    ## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
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


class IGMoCEIR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=32,
                 levels=4,
                 heads=[1, 1, 1, 1],
                 num_blocks=[1, 1, 1, 3],
                 num_dec_blocks=[1, 1, 1],
                 ffn_expansion_factor=2,
                 LayerNorm_type='WithBias',  # 'WithBias' or 'BiasFree'
                 DW_Expand=2,
                 dilations=[1,4,9],
                 fusion_weight=0.5,
                 bias=False,
                 rank=2,
                 num_experts=4,
                 depth_type="lin",
                 stage_depth=[3, 2, 1],
                 rank_type="constant",
                 topk=1,
                 expert_layer=FFTAttention,
                 with_complexity=False,
                 complexity_scale="max",
                 ):
        super(IGMoCEIR, self).__init__()

        self.dim = dim
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.padder_size = 2 ** len(num_blocks)

        self.estimator = Illu_Estimator(layers=3,dim=dim)

        dims = [dim * 2 ** i for i in range(levels)]
        ranks = [rank for i in range(levels - 1)]

        self.patch_embed = OverlapPatchEmbed(in_c=in_channels, embed_dim=dim, bias=False)
        self.denoise_1 = Denoise_1(dim)
        self.illu_patch_embed = nn.Conv2d(3, dim//2, 3, 1, 1)

        self.illu_downsamples = nn.ModuleList([
            Downsample(dim // 2 * 2 ** i) for i in range(levels - 1)
        ])

        # Encoder
        self.enc = nn.ModuleList([])
        for i in range(levels - 1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i],
                    num_blocks=num_blocks[i],
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type,
                    bias=True,
                    DW_Expand=DW_Expand,
                    dilations=dilations,
                    fusion_weight=fusion_weight,
                    ),
                Downsample(dim * 2 ** i)
            ])
            )
        # Latent
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1],
            num_heads=heads[-1],
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,
            DW_Expand=DW_Expand,
            dilations=dilations,
            fusion_weight=fusion_weight,
        )
        # -- Decoder --
        dims = dims[::-1]
        ranks = ranks[::-1]
        heads = heads[::-1]
        num_dec_blocks = num_dec_blocks[::-1]

        self.dec = nn.ModuleList([])
        for i in range(levels - 1):
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, bias=bias),
                DecoderResidualGroup(
                    dim=dims[i + 1],
                    num_blocks=num_dec_blocks[i],
                    num_heads=heads[i + 1],
                    ffn_expansion=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type, bias=bias, expert_layer=expert_layer, freq_dim=dims[0],
                    with_complexity=with_complexity,
                    rank=ranks[i], num_experts=num_experts, stage_depth=stage_depth[i], depth_type=depth_type,
                    rank_type=rank_type, top_k=topk, complexity_scale=complexity_scale),
            ])
            )

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)  # 卷积输出层
        self.total_loss = None

    def forward(self, img,label=None):
        shortcut = img
        x = self.check_image_size(img)
        eps = 1e-6
        input = x + eps

        L2 = input - self.denoise_1(input)

        illu_map = self.estimator(L2.detach())

        illu_fea = self.illu_patch_embed(illu_map)


        feats = self.patch_embed(x)

        self.total_loss = 0
        enc_feats = []

        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats, illu_fea)
            enc_feats.append(feats)
            feats = downsample(feats)
            illu_fea = self.illu_downsamples[i](illu_fea)

        feats = self.latent(feats, illu_fea)

        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            feats = block(feats)
            self.total_loss += block.loss

        # feats = self.refinement(feats)
        out = self.output(feats) + shortcut

        self.total_loss /= sum(self.num_dec_blocks)
        return out


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # 需要填充的高度和宽度，以确保图像的高度和宽度是 self.padder_size 的倍数。
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)  # 使用F.pad函数填充图像。
        return x

if __name__ == '__main__':
    # img = torch.randn(1, 3, 256, 256)
    # model = Model(dim=32, num_heads=8, ffn_expansion_factor=4,
    #                      bias=True,LayerNorm_type='WithBias', DW_Expand=2,
    #                      dilations=[1, 4, 9], fusion_weight=0.5)
    model = IGMoCEIR(rank=2, num_blocks=[4,6,6,8],
                  num_dec_blocks=[2, 4, 4],
                  dim=32,
                  levels=4,
                  with_complexity=True,
                  complexity_scale="max",
                  stage_depth=[1, 1, 1],
                  depth_type="constant",
                  rank_type="spread",
                  num_experts=4,
                  topk=1,
                  expert_layer=FFTAttention,
                  DW_Expand=2,
                  dilations=[1, 2, 3], fusion_weight=0.5, bias=True).cuda()
    # 计算模型参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {model_params / 1e6:.3f} M")

    x = torch.randn(1, 3, 224, 224).cuda()
    _ = model(x)
    print(model.total_loss)
    # Memory usage
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery',
                                         torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2))

    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))
