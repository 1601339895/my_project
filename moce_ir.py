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
    将模块的参数清零并返回。
    """
    for p in module.parameters():  # 遍历模块的所有参数
        p.detach().zero_()  # 将参数值置零
    return module


class MySequential(nn.Sequential):
    """
    重写了nn.Sequential容器中的forward函数，使其可以接受两个输入，并将两个输入分别传递给每一层。
    """

    def forward(self, x1, x2):
        # Iterate through all layers in sequential order
        # 按顺序遍历所有层
        for layer in self:  # self是一个包含多个层的列表
            # Check if the layer takes two inputs (i.e., custom layers)
            # 检查该层是否接受两个输入（即自定义层）
            if isinstance(layer, nn.Module):  # 检查该层是否为nn.Module类型，则调用该层的forward函数，并传入两个输入
                # Pass both inputs to the layer
                # 将两个输入传递到层
                x1 = layer(x1, x2)
            else:
                # For non-module layers, pass the two inputs directly
                # 对于非模块层，直接传递两个输入
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
    """
    用于将输入数据分配到不同的专家中，并根据门的权重来将专家输出组合成最终输出。
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates  # 存储门值矩阵gates，shape为(batch_size, num_experts),其中gates[b, i]表示批次b中的样本对专家的权重
        self._num_experts = num_experts  # 存储专家数量num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)  # 找出路由的非零值的位置，并按照专家索引进行排序
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)  # 按照专家索引分割gates，得到每个样本对应的专家索引
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]  # 根据排序后的索引，得到每个专家对应的批次索引
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()  # 计算每个专家负责的样本数量
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]  # 根据批次索引，扩展gates，使其与输入的shape相同
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)  # 根据专家索引，得到每个样本对应的非零门值

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        为每个专家创建一个输入张量。专家“i”的“张量”包含与批次元素“b”对应的“inp”切片，其中“gates[b, i] > 0”。
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  # inp为输入的特征图，扩展为(batch_size, c, h, w)，squeeze(1)去除不必要的维度
        return torch.split(inp_exp, self._part_sizes, dim=0)  # 根据专家负责的样本数量，将输入张量切分为多个张量，每个张量对应一个专家

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        将专家输出相加，并按门控加权。对应于特定批次元素 `b` 的切片计算为专家输出的所有专家 `i` 的总和，并按
        相应的门控值加权。如果 `multiply_by_gates` 设置为 False，则门控值将被忽略
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)  # expert_out为专家输出的列表，stitched为将所有专家输出拼接在一起的大张量

        if multiply_by_gates:  # 如果multiply_by_gates为True，则将门控值乘以专家输出
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)  # 定义一个全0张量，用于将专家输出填充到正确的位置
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())  # 将加权后的专家输出组合在一起，得到最终的输出特征图
        return combined

    def to_spatial(self, x, x_shape):
        h, w = x_shape  # 输入张量的空间维度，h为高，w为宽
        amp, phase = x.chunk(2, dim=1)  # 输入张量在第一维度上分割为振幅和相位
        real = amp * torch.cos(phase)  # 计算实部
        imag = amp * torch.sin(phase)  # 计算虚部
        x = real + 1j * imag  # 复数形式张量
        x = torch.fft.ifft2(x, s=(h, w), norm="backward").real  # 对复数张量进行逆快速傅里叶变换(IFFT)，得到空间域的输出
        return x

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        与每个专家“张量”中的示例相对应的门值。
        """
        # split nonzero gates for each expert
        # 为每个专家拆分非零门值
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
        # 根据 self._part_sizes 将门值矩阵 _nonzero_gates 分割成多个子张量，每个子张量对应一个专家的门值。


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')  # 重塑为3D张量


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # 重塑为4D张量


class BiasFree_LayerNorm(nn.Module):
    """
    无偏置归一化：适用于样本量较小时，此时无偏估计和有偏估计的差异较大，因此可以用无偏置的归一化来提高精度。
    使用N-1的无偏估计来计算方差更能准确反映总体分布，因此在训练时使用N-1的无偏估计来计算方差。
    无偏置的层归一化层。它的主要功能是对输入张量在最后一个维度上进行归一化，
    并通过一个可训练的权重参数进行缩放。该层不包含偏置参数，
    因此它只能调整归一化后的输出的缩放，而不能进行平移
    """

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):  # 判断输入是否为整数
            normalized_shape = (normalized_shape,)  # 如果是整数，则转换为元组
        normalized_shape = torch.Size(normalized_shape)  # 转换为tensor对象，PyTorch中用于表示张量大小的数据类型

        assert len(normalized_shape) == 1  # 输入维度只能为1，层归一化通常只在最后一个维度上进行

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 定义了一个和输入维度相同的权重参数，初始化为1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算输入张量x在最后一个维度上的方差，keepdim=True表示输入和输出保持维度一致，
        # unbiased=False表示计算样本方差，而不是无偏估计
        return x / torch.sqrt(sigma + 1e-5) * self.weight
        # 计算归一化后的输出，1e-5是为了防止sigma为0而导致除零错误，并乘以权重参数


class WithBias_LayerNorm(nn.Module):
    """
    有偏置归一化：适用于样本量较大时，此时有偏估计和无偏估计的差异较小，因此可以用有偏置的归一化来提高精度。
    有偏置的层归一化层。它的主要功能是对输入张量在最后一个维度上进行归一化，
    并通过一个可训练的权重参数进行缩放，并通过一个可训练的偏置参数进行平移。
    """

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):  # 判断归一化维度是否为整数
            normalized_shape = (normalized_shape,)  # 如果是整数，则转换为元组
        normalized_shape = torch.Size(normalized_shape)  # 转换为tensor对象，PyTorch中用于表示张量大小的数据类型

        assert len(normalized_shape) == 1  # 输入维度只能为1，层归一化通常只在最后一个维度上进行

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 定义了一个和输入维度相同的权重参数，初始化为1
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 定义了一个和输入维度相同的偏置参数，初始化为0
        self.normalized_shape = normalized_shape  # 归一化维度

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)  # 计算输入张量x在最后一个维度上的均值，keepdim=True表示输入和输出保持维度一致
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算输入张量x在最后一个维度上的方差，keepdim=True表示输入和输出保持维度一致，
        # unbiased=False表示计算样本方差，而不是无偏估计
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        # x - mu 平移，/sqrt(sigma + 1e-5) 缩放，*self.weight 缩放，+self.bias 平移，得到归一化后的输出


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim  # 通道数
        if LayerNorm_type == 'BiasFree':  # 无偏置的层归一化
            self.body = BiasFree_LayerNorm(dim)  # 调用无偏置层归一化
        else:
            self.body = WithBias_LayerNorm(dim)  # 调用有偏置层归一化

    def forward(self, x):
        h, w = x.shape[-2:]  # 输入图像的高和宽
        return to_4d(self.body(to_3d(x)), h, w)  # 调用层归一化函数，并转换为4D张量


class HighPassConv2d(nn.Module):
    def __init__(self, c, freeze):
        super().__init__()
        """
       HighPassConv2d 模块的主要功能是对输入图像的每一个通道应用一个 3x3 的高通滤波器卷积核。
       高通滤波器的作用是突出图像中的高频细节，同时削弱低频噪声
        """
        self.conv = nn.Conv2d(
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=c
        )  # groups=c 使得每一个通道都参与卷积，而不是每个通道一个卷积核

        kernel = torch.tensor([[[[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]]]], dtype=torch.float32)  # 定义卷积核
        self.conv.weight.data = kernel.repeat(c, 1, 1, 1)  # 重复c次卷积核

        if freeze:  # 冻结权重
            self.conv.requires_grad_ = False

    def forward(self, x):
        return self.conv(x)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
##门控深度可分离卷积前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        """
        :param dim: 输入特征图的维度
        :param ffn_expansion_factor: 前馈网络的扩张因子，用于控制中间隐藏层的维度
        :param bias: 是否使用偏置
        """
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)  # 根据输入特征图的维度和扩张因子计算中间隐藏层的维度，
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1,
                                    bias=bias)  # 定义一个1x1卷积层，用于维度扩张，输出维度为2倍的输入特征图维度
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)  # 定义一个3x3深度可分离卷积层，用于特征提取
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)  # 1x1卷积层，用于维度缩减，映射回输入特征图的维度

    def forward(self, x):
        x = self.project_in(x)  # 扩张维度，输出维度为2倍的输入特征图维度
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 特征提取，将输入特征图分成两个部分，分别进行3x3深度可分离卷积，得到两个特征图
        x = F.gelu(x1) * x2  # x1通过gelu激活函数与x2相乘，用于控制信息流通，门控机制
        x = self.project_out(x)  # 缩减维度，映射回输入特征图的维度
        return x

    ##########################################################################


## Multi-DConv Head Transposed Self-Attention
## 多DConv头转置自注意力机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        """
        self-attention模块
        :param dim: 输入特征图的维度
        :param num_heads: 自注意力头数
        :param bias: 偏置
        """
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 定义一个可训练的参数矩阵，用于控制注意力的强度，shape为(num_heads, 1, 1)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 定义qkv卷积层, 输入通道数为dim, 输出通道数为dim*3, 卷积核大小为1x1
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)  # 定义qkv_dwconv卷积层, 输入通道数为dim*3, 输出通道数为dim*3, 卷积核大小为3x3, 步长为1, 填充为1, 组数为dim*3
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 定义输出卷积层, 输入通道数为dim, 输出通道数为dim, 卷积核大小为1x1

    def forward(self, x):
        b, c, h, w = x.shape  # 输入张量的batch大小，通道数，高，宽

        qkv = self.qkv_dwconv(self.qkv(x))  # 先进行qkv卷积映射到Q,K,V的特征空间，再进行dwconv卷积
        q, k, v = qkv.chunk(3, dim=1)  # 将处理后的Q,K,V张量分割为q,k,v，得到3个独立张量，每个张量的通道数为dim

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 重塑Q特征空间
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 重塑K特征空间
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 重塑V特征空间

        q = torch.nn.functional.normalize(q, dim=-1)  # 归一化Q特征空间，目的是为了计算注意力权重时保持数值稳定
        k = torch.nn.functional.normalize(k, dim=-1)  # 归一化K特征空间，目的是为了计算注意力权重时保持数值稳定

        attn = (q @ k.transpose(-2,-1)) * self.temperature  # 计算注意力权重，Q和转置后的K矩阵相乘，并乘以一个可训练的参数矩阵temperature，得到注意力权重矩阵attn
        attn = attn.softmax(dim=-1)  # 计算注意力权重矩阵的softmax归一化

        out = (attn @ v)  # 计算注意力输出,自注意力机制的输出

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # 将输出张量转换为原来的4D张量

        out = self.project_out(out)  # 通过1x1卷积层输出特征图
        return out  # shape: (b, c, h, w) -> [1,3,224,224]



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        """
        交叉注意力模块
        :param dim:输入特征图的维度
        :param num_heads: 自注意力头数
        :param bias: 是否使用偏置
        """
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads  # 自注意力头数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 定义一个可训练的参数矩阵，用于控制注意力的强度，shape为(num_heads, 1, 1)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 定义q卷积层, 输入通道数为dim, 输出通道数为dim, 卷积核大小为1x1
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,
                                  bias=bias)  # 定义q_dwconv深度可分离卷积层, 输入通道数为dim, 输出通道数为dim, 卷积核大小为3x3, 步长为1, 填充为1, 组数为dim
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)  # 定义kv卷积层, 输入通道数为dim, 输出通道数为dim*2, 卷积核大小为1x1
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, stride=1, padding=7 // 2, groups=dim * 2,
                                   bias=bias)  # 定义kv_dwconv深度可分离卷积层, 输入通道数为dim*2, 输出通道数为dim*2, 卷积核大小为7x7, 步长为1, 填充为3, 组数为dim*2
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 定义输出卷积层, 输入通道数为dim, 输出通道数为dim, 卷积核大小为1x1

    def forward(self, x, y):
        b, c, h, w = x.shape  # 输入张量的batch大小，通道数，高，宽

        q = self.q_dwconv(self.q(x))  # 先对x进行q卷积映射到Q的特征空间，再进行dwconv卷积
        kv = self.kv_dwconv(self.kv(y))  # 先对y进行kv卷积映射到K,V的特征空间，再进行dwconv卷积
        k, v = kv.chunk(2, dim=1)  # 将kv特征空间分成K,V两个部分

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)  # 归一化Q特征空间，目的是为了计算注意力权重时保持数值稳定
        k = torch.nn.functional.normalize(k, dim=-1)  # 归一化K特征空间，目的是为了计算注意力权重时保持数值稳定

        attn = (q @ k.transpose(-2,
                                -1)) * self.temperature  # 计算注意力权重，Q和转置后的K矩阵相乘，并乘以一个可训练的参数矩阵temperature，得到注意力权重矩阵attn
        attn = attn.softmax(dim=-1)  # 计算注意力权重矩阵的softmax归一化

        out = (attn @ v)  # 计算注意力输出,交叉注意力机制的输出

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # 将输出张量转换为原来的4D张量

        out = self.project_out(out)  # 通过1x1卷积层输出特征图
        return out


##########################################################################
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

"""####################################################"""
# patch级别
"""####################################################"""
class DualGranularityAdapter(nn.Module):
    def __init__(self, dim, rank, num_experts=4, top_k=2, expert_layer=FFTAttention, stage_depth=1,
                 depth_type="constant", rank_type="constant", freq_dim=128, patch_size=8,
                 with_complexity=False, complexity_scale="min"):
        super().__init__()
        self.global_adapter = AdapterLayer(
            dim, rank, num_experts=num_experts, top_k=top_k,
            expert_layer=expert_layer, stage_depth=stage_depth,
            depth_type=depth_type, rank_type=rank_type, freq_dim=freq_dim,
            with_complexity=with_complexity, complexity_scale=complexity_scale
        )

        self.patch_adapter = PatchAdapterLayer(
            dim, rank, num_experts=num_experts, top_k=1,
            expert_layer=expert_layer, stage_depth=1,
            patch_size=patch_size, depth_type=depth_type, rank_type=rank_type
        )

    def forward(self, x, freq_emb, shared):
        x = self.global_adapter(x, freq_emb, shared)     # 全局专家
        x = self.patch_adapter(x, shared)                # Patch专家
        return x

class PatchAdapterLayer(nn.Module):
    def __init__(self, dim, rank, num_experts=4, top_k=1,
                 patch_size=8, expert_layer=FFTAttention, stage_depth=1,
                 depth_type="constant", rank_type="constant"):
        super(PatchAdapterLayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.patch_size = patch_size

        self.router = PatchRoutingFunction(dim, num_experts, patch_size=patch_size, k=top_k)

        if depth_type == "constant":
            depths = [stage_depth] * num_experts
        if rank_type == "constant":
            ranks = [rank] * num_experts
        kernel_sizes = [3 + 2 * i for i in range(num_experts)]

        self.experts = nn.ModuleList([
            PatchExpert(dim, ranks[i], expert_layer, depths[i], patch_size, kernel_sizes[i])
            for i in range(num_experts)
        ])

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, shared):
        gates, indices, values = self.router(x)  # [B, E, H, W]
        outputs = []
        for i, expert in enumerate(self.experts):
            mask = gates[:, i:i+1, :, :]  # [B,1,H,W]
            if mask.sum() > 0:
                masked_x = x * mask
                masked_shared = shared * mask
                out = expert(masked_x, masked_shared)
            else:
                out = torch.zeros_like(x)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # [B, E, C, H, W]
        weighted_outputs = (gates.unsqueeze(2) * outputs).sum(dim=1)  # [B, C, H, W]
        return self.proj_out(weighted_outputs)

class PatchRoutingFunction(nn.Module):
    def __init__(self, dim, num_experts, patch_size=8, k=1):
        super(PatchRoutingFunction, self).__init__()
        self.k = k
        self.patch_size = patch_size
        self.num_experts = num_experts
        self.router = nn.Conv2d(dim, num_experts, kernel_size=1)

    def forward(self, x):
        logits = self.router(x)  # [B, E, H, W]
        probs = F.softmax(logits, dim=1)
        top_k_values, top_k_indices = torch.topk(probs, self.k, dim=1)  # [B, k, H, W]
        gates = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values

class PatchExpert(nn.Module):
    def __init__(self, dim, rank, func, depth, patch_size, kernel_size):
        super(PatchExpert, self).__init__()
        self.expert = ModExpert(dim, rank, func, depth, patch_size, kernel_size)

    def forward(self, x, shared):
        return self.expert(x, shared)


"""####################################################"""
"""####################################################"""

##########################################################################
## Adapter Block    
class ModExpert(nn.Module):
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size: int):
        """
        专家模块，包含多个层，每个层包含多个专家。
        :param dim: 输入特征图的维度
        :param rank: 秩数
        :param func: 具体的专家层类型，例如FFTAttention
        :param depth: 表示专家模块的层数
        :param patch_size: 用于操作的patch大小
        :param kernel_size: 卷积核大小
        """
        super(ModExpert, self).__init__()

        self.depth = depth  # 专家模块的层数
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        # 第一个卷积层用于将输入特征图映射到秩空间
        # 第二个卷积层用于门控机制，将输入特征图映射到秩空间，用于门控机制
        # 第三个卷积层用于将秩空间映射回输入特征图
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)  # 具体使用的专家层类型实例化后的对象，
        # 例如FFTAttention(rank, kernel_size=kernel_size, patch_size=patch_size)，func是一个类，在process函数中被调用，

    def process(self, x, shared):
        """
        专家模块的处理函数，将输入特征图和共享特征图输入到专家层，并输出处理后的特征图。
        :param x: 输入特征图
        :param shared: 共享特征图
        :return: 原始输入特征图和处理后的特征图的和
        """
        shortcut = x  # 保存原始输入特征图用于残差连接
        x = self.proj[0](x)  # 首先输入特征图通过投影层的第一个卷积层映射到秩空间
        x = self.body(x) * F.silu(self.proj[1](shared))
        # 将上述处理后的特征图应用body专家层，
        # 同时将共享特征图通过投影层的第二个卷积层映射到秩空间，再经过silu激活函数作为门控信号，并与body专家层的输出相乘，
        x = self.proj[2](x)  # 将上述处理后的x通过投影层的第三个卷积层映射回输入特征图
        return x + shortcut

    def feat_extract(self, feats, shared):
        """
        专家模块的特征提取函数，将输入特征图和共享特征图输入到专家层，并输出处理后的特征图。
        :param feats: 输入特征图
        :param shared: 共享特征图
        :return: 处理后的特征图
        """
        for _ in range(self.depth):
            feat = self.process(feats, shared)
        # 输入的特征图和共享特征图会经过self.depth次的process函数处理，并将处理后的特征图叠加到一起，
        return feat

    def forward(self, x, shared):
        b, c, h, w = x.shape

        if b == 0:  # 输入特征图的batch大小为0，则直接返回
            return x
        else:
            x = self.feat_extract(x, shared)
            return x


########################################################################### 
## Adapter Layer
class AdapterLayer(nn.Module):
    def __init__(self,
                 dim: int, rank: int, num_experts: int = 4, top_k: int = 2, expert_layer: nn.Module = FFTAttention,
                 stage_depth: int = 1,
                 depth_type: str = "lin", rank_type: str = "constant", freq_dim: int = 128,
                 with_complexity: bool = False, complexity_scale: str = "min"):
        """
        自适应层，包含多个专家模块，每个专家模块包含多个层，每个层包含多个专家。
        :param dim:  输入特征图的维度
        :param rank: 用于频域嵌入的秩
        :param num_experts:专家数量，默认为4
        :param top_k: 选择的专家数量，默认为2
        :param expert_layer: 专家层的类型，默认为FFTAttention
        :param stage_depth: 专家层的深度，默认为1
        :param depth_type: 深度类型，默认为lin
        :param rank_type: 秩类型，默认为constant
        :param freq_dim: 频域嵌入的维度，默认为128
        :param with_complexity: 是否考虑专家复杂度，默认为False
        :param complexity_scale: 专家复杂度的缩放方式，默认为min
        """
        super().__init__()

        self.tau = 1  # 缩放因子，用于控制重要性损失，
        self.loss = None  # 存储辅助损失
        self.top_k = top_k  # 选择的专家数量
        self.noise_eps = 1e-2  # 噪声的标准差
        self.num_experts = num_experts  # 专家数量

        patch_sizes = [2 ** (i + 2) for i in range(num_experts)]  # 计算专家层的patch大小，依次为4,8,16,32
        kernel_sizes = [3 + (2 * i) for i in range(num_experts)]  # 计算专家层的卷积核大小，依次为3,5,7,9

        # 计算专家层的深度
        if depth_type == "lin":  # 线性深度
            depths = [stage_depth + i for i in range(num_experts)]  # 线性深度递增
        elif depth_type == "double":  # 双倍深度
            depths = [stage_depth + (2 * i) for i in range(num_experts)]  # 双倍深度递增
        elif depth_type == "exp":  # 指数深度
            depths = [2 ** (i) for i in range(num_experts)]  # 指数深度递增
        elif depth_type == "fact":  # 阶乘深度
            depths = [math.factorial(i + 1) for i in range(num_experts)]  # 阶乘深度递增
        elif isinstance(depth_type, int):  # 固定深度
            depths = [depth_type for _ in range(num_experts)]  # 固定深度
        elif depth_type == "constant":  # 常数深度
            depths = [stage_depth for i in range(num_experts)]  # 常数深度
        else:  # 未知深度类型
            raise NotImplementedError  # 触发未知深度类型错误

        if rank_type == "constant":  # 常数秩
            ranks = [rank for _ in range(num_experts)]
        elif rank_type == "lin":  # 线性秩
            ranks = [rank + i for i in range(num_experts)]  # 线性秩递增
        elif rank_type == "double":  # 双倍秩
            ranks = [rank + (2 * i) for i in range(num_experts)]  # 双倍秩递增
        elif rank_type == "exp":  # 指数秩
            ranks = [rank ** (i + 1) for i in range(num_experts)]  # 指数秩递增
        elif rank_type == "fact":  # 阶乘秩
            ranks = [math.factorial(rank + i) for i in range(num_experts)]  # 阶乘秩递增
        elif rank_type == "spread":  # 均匀秩
            ranks = [dim // (2 ** i) for i in range(num_experts)][::-1]  # 均匀秩递减
        else:  # 未知秩类型
            raise NotImplementedError  # 触发未知秩类型错误

        self.experts = nn.ModuleList([
            MySequential(
                *[ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel)])
            for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
        ])  # 专家模块列表，包含多个专家模块，每个专家模块包含多个层，每个层包含多个专家。

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)  # 1x1卷积层，用于输出特征图的投影
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])  # 计算专家复杂度
        self.routing = RoutingFunction(
            dim, freq_dim,
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale
        )  # 路由函数，用于选择专家

    def forward(self, x, freq_emb, shared):
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x,
                                                                    freq_emb)  # 通过路由函数返回选择的专家矩阵、选择的专家的索引、选择的专家的评分、辅助损失
        self.loss = aux_loss  # 存储辅助损失

        # routing
        if self.training:  # 如果是训练模型，则调用稀疏调度器SparseDispatcher，将输入特征图x和共享特征图shared分别调度到各个专家模块
            dispatcher = SparseDispatcher(self.num_experts, gates)  # 创建稀疏调度器
            expert_inputs = dispatcher.dispatch(x)  # 将输入特征图调度到各个专家模块
            expert_shared_intputs = dispatcher.dispatch(shared)  # 将共享特征图调度到各个专家模块
            expert_outputs = [self.experts[exp](expert_inputs[exp], expert_shared_intputs[exp]) for exp in
                              range(len(self.experts))]  # 输入到各个专家模块，得到各个专家的输出
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:  # 测试阶段
            selected_experts = [self.experts[i] for i in top_k_indices.squeeze(0)]  # Select the corresponding experts
            expert_outputs = torch.stack([expert(x, shared) for expert in selected_experts], dim=1)
            gates = gates.gather(1, top_k_indices)
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs
            out = weighted_outputs.sum(dim=1)  # Sum across the top-k dimension to get the final output

        out = self.proj_out(out)  # 输出特征图的投影
        return out


class RoutingFunction(nn.Module):
    """
    计算路由函数，用于选择专家。
    """

    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias: bool = True,
                 complexity_scale: str = "max"):
        """
        初始化路由函数
        :param dim:输入特征图的维度
        :param freq_dim: 频域嵌入的维度
        :param num_experts: 专家数量
        :param k: 选择的专家数量
        :param complexity: 每个专家的复杂度
        :param use_complexity_bias: 是否使用复杂度偏置，默认为True
        :param complexity_scale: 复杂度缩放方式，默认为max
        """
        super(RoutingFunction, self).__init__()

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        )
        # 门层，首先对对输入特征图进行全局平均池化，将特征图的尺寸缩减为1x1，
        # 将形状进行重排，得到[b,c]的向量，再输入线性层，得到[b,num_experts]的输出，每行代表一个输入样本对各个专家的评分

        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)  # 频域门层，将频域嵌入(freq_dim维)投影到专家评分(num_experts维)
        if complexity_scale == "min":  # 归一化后，最小值会变为 1，其他值会大于或等于 1。
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":  # 归一化后，最大值会变为 1，其他值会小于或等于 1。
            complexity = complexity / complexity.max()
        # 对复杂度权重complexity进行归一化处理。
        # 归一化的目的是将复杂度权重缩放到一个特定的范围（例如 [0, 1]），以便在后续计算中更方便地使用
        # 根据complexity_scale参数来缩放每个专家的复杂度。
        # 如果设置为"max"，则复杂度值会被除以其最大值；如果设置为"min"，则复杂度值会被除以其最小值
        # 作用是1.平衡不同专家的复杂度，不同专家可能具有不同的复杂度（例如计算量或参数量）。
        # 如果直接使用原始复杂度值，可能会导致某些专家的权重过高或过低，从而影响模型的训练效果。
        # 2.控制辅助损失的尺度。3.灵活性

        self.register_buffer('complexity', complexity)
        # register_buffer是Pyorch中nn.Module的函数，用于将张量注册为模型的一部分，但不会参与梯度计算(不会参与训练)。
        # 这里的 complexity 是一个张量，表示每个专家的复杂度权重。
        # 通过 register_buffer，这个张量被注册为模型的一部分，但不会参与梯度计算。
        # 在模型保存、加载或移动设备时，complexity 会自动被处理。

        self.k = k  # 选择的专家数量
        self.tau = 1  # 温度参数，用于控制 softmax 的分布。
        self.num_experts = num_experts  # 专家数量
        self.noise_std = (1.0 / num_experts) * 1.0  # 噪声的标准差
        self.use_complexity_bias = use_complexity_bias  # 是否使用复杂度偏置

    def forward(self, x, freq_emb):
        logits = self.gate(x) + self.freq_gate(freq_emb)  # 计算每个输入样本对各个专家的评分logits
        if self.training:  # 在训练模式下，计算重要性损失（importance loss），衡量每个专家被选择的重要性。
            loss_imp = self.importance_loss(logits.softmax(dim=-1))

        noise = torch.randn_like(logits) * self.noise_std  # 添加噪声
        noisy_logits = logits + noise  # 加噪声后的logits
        gating_scores = noisy_logits.softmax(dim=-1)  # 通过softmax归一化后的得分
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)  # 使用 torch.topk 选出得分最高的 k 个专家及其索引。

        # Final auxiliary loss
        if self.training:  # 在训练模式下，计算辅助损失（auxiliary loss），用于控制专家的负载
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)  # 计算负载损失，衡量每个专家的负载均衡的情况
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load  #
        else:
            aux_loss = 0

        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)  # gates 是一个稀疏矩阵，仅在选中的专家位置有非零值。
        # 例如，如果 top_k_indices 值为 [0, 2, 1]，则 gates 值为 [[1, 0, 0], [0, 0, 1], [0, 1, 0]]。
        return gates, top_k_indices, top_k_values, aux_loss
        # 返回 gates 矩阵，表示每个输入样本对各个专家的选择概率，top_k_indices 矩阵，表示每个输入样本选择的 k 个专家的索引，
        # top_k_values 矩阵，表示每个输入样本选择的 k 个专家的评分，aux_loss 辅助损失。

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)  # 计算每个专家的重要性得分
        importance = importance * (
                self.complexity * self.tau) if self.use_complexity_bias else importance  # 判断是否使用复杂度偏置
        # 如果使用复杂度偏置，则将复杂度权重乘以 tau 系数，并与重要性得分相乘。
        # 否则，只使用重要性得分。
        imp_mean = importance.mean()  # 计算重要性得分的均值
        imp_std = importance.std()  # 计算重要性得分的标准差
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2  # 计算重要性损失变异系数的平方作为损失。

        return loss_imp

    def load_loss(self, logits, logits_noisy, noise_std):
        # Compute the noise threshold
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]  # 找到第 k 大的得分作为阈值。

        # Compute the load for each expert
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )  # 计算每个样本的阈值对应的专家得分。

        # Calculate noise required to win
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        # 计算每个样本需要多少噪声才能超过阈值。

        # Probability of being above the threshold
        # 使用正态分布计算每个专家超过阈值的概率。
        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)

        # Compute mean probability for each expert over examples
        p_mean = p.mean(dim=0)

        # Compute p_mean's coefficient of variation squared
        # 计算概率的均值和标准差，并返回其变异系数的平方作为损失。
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2

        return loss_load


##########################################################################
## Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        """
        编码器块
        :param dim: 输入特征图的通道数
        :param num_heads: 多头注意力头数
        :param ffn_expansion_factor: 前馈网络的扩张因子
        :param bias:偏置
        :param LayerNorm_type:层归一化类型
        """
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type)
        ])  # 两个归一化层

        self.mixer = Attention(dim, num_heads, bias)  # 注意力层
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)  # 前馈网络层

    def forward(self, x):
        x = x + self.mixer(self.norms[0](x))
        # x首先经过第一个归一化层，目的是减小内部协变量偏移，然后经过注意力层，使得每一层的输入分布更加稳定，从而加速训练过程。
        # 经过归一化后的x传递给自注意力机制，作用是输入特征图的不同位置之间学习依赖性，通过加权求和来生成每个位置的新表示。
        # 最后加上残差连接，将原始x与经过注意力机制处理后的x相加，有助于梯度传播，缓解梯度消失问题。
        x = x + self.ffn(self.norms[1](x))
        # 对上面的x再次应用层归一化，确保输入特征图在经过前馈网络之前具有稳定分布。
        # 然后将经过前馈网络处理后的x与原始x相加，作为下一个编码器块的输入。
        # 最后经过前馈网络的输出与经过注意力机制和第一个归一化层处理后的x相加，使用第二次残差连接。
        return x


##########################################################################
## Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, expert_layer, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, depth_type=None, rank_type=None, stage_depth=None,
                 freq_dim: int = 128, with_complexity: bool = False):
        """
        解码器块
        :param dim: 输入特征图的通道数
        :param num_heads: 自注意力头数
        :param ffn_expansion_factor: FFN扩张因子
        :param bias: 是否使用偏置
        :param LayerNorm_type: 层归一化类型
        :param expert_layer: 专家层类型
        :param complexity_scale: 复杂度缩放类型
        :param rank: 用于频域嵌入的秩
        :param num_experts: 专家数量
        :param top_k: 选择的专家数量
        :param depth_type: 深度类型
        :param rank_type: 秩类型
        :param stage_depth: 每个阶段的深度
        :param freq_dim: 频域嵌入的维度，默认值为128
        :param with_complexity: 是否使用复杂度偏置
        """
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type),
        ])
        # 两个归一化层，用于在特征图的最后一个维度上进行归一化操作，这两个层归一化的类型和维度都相同。

        self.proj = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        ])
        # 投影层，用于将输入特征图投影到与输出特征图相同的维度。

        self.shared = Attention(dim, num_heads, bias)  # 自注意力层
        self.mixer = CrossAttention(dim, num_heads=num_heads, bias=bias)  # 交叉注意力层，用于在两个特征图之间秩序交叉注意力。
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)  # 前馈网络层

        self.adapter = AdapterLayer(
            dim, rank,
            top_k=top_k, num_experts=num_experts, expert_layer=expert_layer, freq_dim=freq_dim,
            depth_type=depth_type, rank_type=rank_type, stage_depth=stage_depth,
            with_complexity=with_complexity, complexity_scale=complexity_scale
        )  # 适配层，用于将输入特征图适配到专家层，包含多个专家，每个专家负责处理不同区域的特征图。

    def forward(self, x, freq_emb=None):
        shortcut = x  # 保存原始输入特征图
        x = self.norms[0](x)  # 经过第一个归一化层
        # 将经过第一个归一化层后的x分为两个特征图x_s和x_a，
        x_s = self.proj[0](x)  # 通过1x1卷积投影到与输出特征图相同的维度。
        x_a = self.proj[1](x)  # 通过1x1卷积投影到与输出特征图相同的维度。
        x_s = self.shared(x_s)  # 经过自注意力层，用于增强特征图内部的空间相关性。
        x_a = self.adapter(x_a, freq_emb, x_s)  # x_a经过适配层，适配器层根据freq_emb和x_s进行专家操作，用于增强特征图的频域信息
        x = self.mixer(x_a, x_s) + shortcut  # 经过交叉注意力层，将x_a和x_s进行融合，学习之间的依赖关系，然后将融合后的特征图与原始x进行残差连接，增强梯度传播。

        x = x + self.ffn(self.norms[1](x))  # 再对融合后的特征图进行一次归一化，然后再经过前馈网络层，增强特征图的非线性表达能力。
        return x, self.adapter.loss


######################################################################
## Encoder Residual Group
## 编码器残差块组
class EncoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool):
        """
        编码器残差块组
        :param dim: 输入特征图的通道数
        :param num_heads: 多头注意力的头数列表
        :param num_blocks: 编码器残差块的数量
        :param ffn_expansion:  FFN扩张因子，用于控制中间隐藏层的维度。
        :param LayerNorm_type: LayerNorm类型
        :param bias: 是否使用偏置
        """
        super().__init__()

        self.loss = None  # 用于存储该编码器残差块组的辅助损失，初始值为N
        self.num_blocks = num_blocks  # 编码器残差块的数量

        self.layers = nn.ModuleList([])  # 编码器残差块列表
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

    ######################################################################


## Decoder Residual Group
## 解码器残差块组
class DecoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool,
                 complexity_scale=None,
                 rank=None, num_experts=None, expert_layer=None, top_k=None, depth_type=None, stage_depth=None,
                 rank_type=None, freq_dim: int = 128, with_complexity: bool = False):
        """
        解码器残差块组
        :param dim:  输入特征图的通道数
        :param num_heads: 多头注意力的头数列表
        :param num_blocks: 解码器残差块组中的解码器数量
        :param ffn_expansion: 前馈网络的扩张因子，用于控制中间隐藏层的维度。
        :param LayerNorm_type: 层归一化类型
        :param bias: 是否在卷积层中使用偏置
        :param complexity_scale: 复杂度缩放类型
        :param rank: 用于频率嵌入的秩
        :param num_experts: 专家层的数量
        :param expert_layer: 专家层的类型
        :param top_k: 选择的专家数量
        :param depth_type: 深度类型
        :param stage_depth: 每个阶段的深度设置
        :param rank_type: 秩类型
        :param freq_dim: 频域嵌入的维度，默认值为128
        :param with_complexity: 是否使用复杂度
        """
        super().__init__()

        self.loss = None  # 存储该解码器残差块组的辅助损失，初始值为None
        self.num_blocks = num_blocks  # 存储解码器块的数量

        self.layers = nn.ModuleList([])  # 存储多个解码器块
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
            x, loss = self.layers[i](x, freq_emb)
            self.loss += loss
            i += 1
        return x

    ##########################################################################


## Overlapped image patch embedding with 3x3 Conv
## 使用 3x3 卷积层嵌入重叠图像块
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        """
        嵌入重叠图像块
        :param in_c: 输入通道数
        :param embed_dim:  嵌入维度
        :param bias:  是否使用偏置
        """
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)  # 3x3卷积层嵌入图像块

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        """
        下采样模块
        :param n_feat:输入特征图的通道数
        """
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
        # 第一个层是3x3卷积层，输入通道数为n_feat，输出通道数为n_feat//2，减为一半，步长为1，填充为1，不使用偏置。
        # 第二层是PixelUnshuffle层，将特征图的高和宽尺寸缩小为原来的一半，缩放因子为2，，通道为维度为输入通道数*缩放因子的平方，
        # 即将特征图的尺寸由[b,c,h,w]变为[b,c*2^2,h//2,w//2]。

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        # 第一个层是3x3卷积层，输入通道数为n_feat，输出通道数为n_feat*2，扩大为两倍，步长为1，填充为1，不使用偏置。
        # 第二层是PixelShuffle层，将特征图的高和宽尺寸扩大为原来的两倍，缩放因子为2，，通道为维度为输入通道数//(缩放因子的平方)，
        # 即将特征图的尺寸由[b,c,h,w]变为[b,c//2^2,h*2,w*2]。

    def forward(self, x):
        return self.body(x)


##########################################################################
## Frequency Embedding
# 频域嵌入模块
class FrequencyEmbedding(nn.Module):
    """
    Embeds magnitude and phase features extracted from the bottleneck of the U-Net.
    从 U-Net 的瓶颈层提取的振幅和相位特征进行嵌入。
    """

    def __init__(self, dim):
        super(FrequencyEmbedding, self).__init__()
        """
        高通滤波卷积层的作用是提取图像的高频信息，这有助于去除图像中的低频噪声
        MLP 用于进一步处理这些高频信息，首先是将特征图的维度增加到 2 * dim，
        然后通过 GELU 激活函数进行非线性变换，最后将维度减少到 dim。
        """
        self.high_conv = nn.Sequential(
            HighPassConv2d(dim, freeze=True),  # 高通滤波器
            nn.GELU())  # gelu激活函数

        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),  # 线性层，输入dim，输出2*dim
            nn.GELU(),  # gelu激活函数
            nn.Linear(2 * dim, dim)  # 线性层，输入2*dim，输出dim
        )

    def forward(self, x):
        x = self.high_conv(x)  # 通过高通滤波器提取频域特征，并进行gelu激活函数，shape为[b,dim,h,w]
        x = x.mean(dim=(-2, -1))  # 全局平均池化，[b,dim,h,w]->[b,dim]，等价于nn.AdaptiveAvgPool2d(1)
        x = self.mlp(x)  # 进行MLP处理，shape为[b,dim]
        return x


##########################################################################
##
class MoCEIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 levels: int = 4,
                 heads=[1, 1, 1, 1],
                 num_blocks=[1, 1, 1, 3],
                 num_dec_blocks=[1, 1, 1],
                 ffn_expansion_factor=2,
                 num_refinement_blocks=1,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
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
        """

        :param inp_channels: 输入通道数(RGB图像)
        :param out_channels: 输出通道数(RGB图像)
        :param dim: 维度
        :param levels: 编解码器的层数
        :param heads: 每一层的多头注意力机制的头数
        :param num_blocks: 每一层的编码器残差组中Transformer块的数量
        :param num_dec_blocks: 每一层的解码器残差组中Transformer块的数量
        :param ffn_expansion_factor: 前馈神经网络的扩张因子
        :param num_refinement_blocks: 精炼层中Transformer块的数量
        :param LayerNorm_type: 层归一化类型，可选'WithBias'或'BiasFree'
        :param bias: 是否在卷积层中使用偏置
        :param rank: 用于频率嵌入的秩
        :param num_experts: 在FFTAttention中使用的专家数量
        :param depth_type: 深度类型
        :param stage_depth: 每个阶段的深度设置
        :param rank_type: 秩类型
        :param topk: 用于选择专家的topk值
        :param expert_layer: 专家层类型
        :param with_complexity: 是否考虑复杂度
        :param complexity_scale: 复杂度缩放类型
        """
        super(MoCEIR, self).__init__()

        self.levels = levels
        self.num_blocks = num_blocks  # num_blocks=[1, 1, 1, 3]意味着第1至第3层各有一个Transformer块，而第4层有三个Transformer块。
        self.num_dec_blocks = num_dec_blocks  # num_dec_blocks=[2, 4, 4]意味着第1至第3层的解码器模块分别有2、4、4个Transformer块
        self.num_refinement_blocks = num_refinement_blocks

        dims = [dim * 2 ** i for i in range(levels)]  # 维度为[32, 64, 128, 256]，levels=4
        ranks = [rank for i in range(levels - 1)]  # 秩为[2, 2, 2],levels=4

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)  # [b,c,h,w] -> [b,dim,h,w]
        self.freq_embed = FrequencyEmbedding(dims[-1])  # [b,dim,h,w] -> [b,dim]

        # -- Encoder --        
        self.enc = nn.ModuleList([])  # 编码器模块列表，包括编码器残差块组和下采样层
        for i in range(levels - 1):  # levels=4,[0,1,2]
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i],
                    num_blocks=num_blocks[i],
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type, bias=True, ),
                Downsample(dim * 2 ** i)
            ])
            )

        # -- Latent --
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1],
            num_heads=heads[-1],
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True, )
        # latent编码器残差块组用于在网络的瓶颈位置对特征图进行深层次的特征提取和变换

        # -- Decoder --
        dims = dims[::-1]  # 维度列表反转，[256, 128, 64, 32]
        ranks = ranks[::-1]  # 秩列表反转，[2, 2, 2]
        heads = heads[::-1]  # heads列表反转，[1, 1, 1, 1]
        num_dec_blocks = num_dec_blocks[::-1]  # num_dec_blocks列表反转，[4, 4, 2]

        self.dec = nn.ModuleList([])  # 解码器模块列表，包括上采样层、
        for i in range(levels - 1):
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, bias=bias),  # 1x1卷积层，用于维度转换，调整通道数
                DecoderResidualGroup(
                    dim=dims[i + 1],
                    num_blocks=num_dec_blocks[i],
                    num_heads=heads[i + 1],
                    ffn_expansion=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type,
                    bias=bias,
                    expert_layer=expert_layer,
                    freq_dim=dims[0],
                    with_complexity=with_complexity,
                    rank=ranks[i],
                    num_experts=num_experts,
                    stage_depth=stage_depth[i],
                    depth_type=depth_type,
                    rank_type=rank_type,
                    top_k=topk,
                    complexity_scale=complexity_scale),
            ])
            )

        # -- Refinement --
        # 精炼层，用于对解码器的输出进行进一步的特征提取和变换
        heads = heads[::-1]  # heads列表反转，为了从解码器到编码器的方向使用多头注意力机制
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks,
            num_heads=heads[0],
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True, )

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)  # 卷积输出层
        self.total_loss = None

    def forward(self, x, labels=None):

        feats = self.patch_embed(x)    # 对输入图像进行重叠图像块嵌入

        self.total_loss = 0  # 总损失
        enc_feats = []  # 编码器特征列表
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)  # 编码器残差块组
            enc_feats.append(feats)
            feats = downsample(feats)  # 下采样层

        feats = self.latent(feats)  # latent编码器残差块组
        freq_emb = self.freq_embed(feats)  # 将经过latent编码器的特征图进行频域嵌入

        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)  # 上采样层
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))  # 特征融合层
            feats = block(feats, freq_emb)  # 解码器残差块组
            self.total_loss += block.loss  # 解码器残差块组的损失累加

        feats = self.refinement(feats)  # 精炼层
        x = self.output(feats) + x  # 将精炼层的特征图通过1x1卷积层调整通道数输出，并与输入图像进行残差连接，得到输出图像

        self.total_loss /= sum(self.num_dec_blocks)  # 计算总的辅助损失的平均值
        return x


if __name__ == "__main__":
    # test
    # model = MoCEIR(rank=2, num_blocks=[4, 6, 6, 8], num_dec_blocks=[2, 4, 4], levels=4, dim=64, num_refinement_blocks=4,
    #                with_complexity=True, complexity_scale="max", stage_depth=[1, 1, 1], depth_type="constant",
    #                rank_type="spread",
    #                num_experts=4, topk=1, expert_layer=FFTAttention).cuda()
    #
    # x = torch.randn(1, 3, 224, 224).cuda()
    # _ = model(x)
    # print(model.total_loss)
    # # Memory usage
    # print('{:>16s} : {:<.3f} [M]'.format('Max Memery',
    #                                      torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2))
    #
    # # FLOPS and PARAMS
    # flops = FlopCountAnalysis(model, (x))
    # print(flop_count_table(flops))
    dim = 32
    levels = 4
    heads = [1, 1, 1, 1]
    num_blocks = [1, 1, 1, 3]
    ffn_expansion_factor = 2
    LayerNorm_type = 'WithBias'
    bias = True

    model = MoCEIR(
        inp_channels=3,
        out_channels=3,
        dim=dim,
        levels=levels,
        heads=heads,
        num_blocks=num_blocks,
        num_dec_blocks=[1, 1, 1],  # 这里需要定义解码器部分的参数以完成实例化
        ffn_expansion_factor=ffn_expansion_factor,
        num_refinement_blocks=1,
        LayerNorm_type=LayerNorm_type,
        bias=bias,
        rank=2,
        num_experts=4,
        depth_type="lin",
        stage_depth=[3, 2, 1],
        rank_type="constant",
        topk=1,
        expert_layer=FFTAttention,
        with_complexity=False,
        complexity_scale="max",
    ).cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    out = model(x)
    print(out.shape)
