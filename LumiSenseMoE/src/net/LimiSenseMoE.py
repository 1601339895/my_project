import math
import numbers
from typing import List

from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torch.nn import functional as F
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


class Illumination_Estimator(nn.Module):
    """
    用于估计光照信息的网络结构
    """

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        """
        :param n_fea_middle: 中间特征图的通道数
        :param n_fea_in: 输入特征图的通道数，默认值为 4
        :param n_fea_out: 输出特征图的通道数，默认值为 3
        """
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1,
                               bias=True)  # 将输入特征图的通道数从n_fea_in变为n_fea_middle，这里使用了1x1的卷积核，即没有空间上的信息变化，只进行通道数的变化。

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)  # 在通道维度上进行卷积，深度可分离卷积。
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1,
                               bias=True)  # 将通道数从n_fea_middle变为n_fea_out，这里使用了1x1的卷积核

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)  # 计算输入图像在每个通道上的均值，并在通道维度上进行扩展，形状为 [b,1,h,w]
        input = torch.cat([img, mean_c], dim=1)  # 将原始输入图像和均值特征拼接在一起，形状为 [b,c+1,h,w]

        x_1 = self.conv1(input)  # 依次通过卷积进行特征提取
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map  # 返回光照特征图，形状为[batch_size, n_fea_middle, h, w]和光照估计图，形状为[batch_size, n_fea_out, h, w]


class IG_MSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 使用卷积层生成 QKV
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dim = dim

    def forward(self, x_in, illu_fea):
        """

        :param x_in: [b,c,h,w]
        :param illu_fea: [b,c,h,w]
        :return: [b,c,h,w]
        """
        b, c, h, w = x_in.shape  # [1,48,224,224]才对
        qkv = self.qkv_dwconv(self.qkv(x_in))  # [1, 144, 224, 224]
        q, k, v = qkv.chunk(3, dim=1)  # [1, 48, 224, 224]

        # 重排 QKV 以适应多头注意力机制
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [1, 1, 48, 50176]
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [1, 1, 48, 50176]
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [1, 1, 48, 50176]
        illu_attn = illu_fea

        illu_attn = rearrange(illu_attn, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [1, 1, 48, 50176]

        v = v * illu_attn  # 将光照注意力放入 V 特征中
        # 归一化 Q 和 K
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 加权求和得到输出
        out = (attn @ v)
        # 恢复输出形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 输出投影
        out = self.project_out(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type)
        ])

        self.mixer = IG_MSA(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, illu_fea):
        x = self.norms[0](x)
        x = x + self.mixer(x, illu_fea)
        x = self.norms[1](x)
        x = x + self.ffn(x)
        return x


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

    def forward(self, x, illu_fea):
        """

        :param x: [b,c,h,w]
        :param illu_fea: [b,c,h,w]
        :return: [b,c,h,w]
        """
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x, illu_fea)
            i += 1
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.estimator = Illumination_Estimator(n_fea_middle=embed_dim)
        # 投影卷积层
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        # 可学习的光照叠加权重
        # self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 光照估计
        _, illu_map = self.estimator(x)
        # 光照图归一化
        # illu_map = (illu_map - illu_map.min()) / (illu_map.max() - illu_map.min() + 1e-5)
        # 光照叠加（带可学习权重）
        # x = x + self.alpha * (x * illu_map)
        # 投影嵌入
        input_img = x * illu_map + x
        out = self.proj(input_img)
        return out


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


########################################
# 解码器
########################################
# Self-Attention in Fourier Domain
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


## Multi-DConv Head Transposed Self-Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

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

    def forward(self, x, freq_emb, shared):
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x, freq_emb)
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


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, expert_layer, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, depth_type=None, rank_type=None, stage_depth=None,
                 freq_dim: int = 128, with_complexity: bool = False):
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

    def forward(self, x, freq_emb=None):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x, loss = self.layers[i](x, freq_emb)
            self.loss += loss
            i += 1
        return x


class LumiSenseMoEIR(nn.Module):
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
        super(LumiSenseMoEIR, self).__init__()

        self.dim = dim
        self.levels = levels
        self.num_blocks = num_blocks  # num_blocks=[1, 1, 1, 3]意味着第1至第3层各有一个Transformer块，而第4层有三个Transformer块。
        self.num_dec_blocks = num_dec_blocks  # num_dec_blocks=[2, 4, 4]意味着第1至第3层的解码器模块分别有2、4、4个Transformer块
        self.num_refinement_blocks = num_refinement_blocks
        self.padder_size = 2 ** len(num_blocks)  # 用于调整输入图像大小的参数，确保它是 2 ** len(num_blocks) 的倍数

        dims = [dim * 2 ** i for i in range(levels)]  # 维度为[32, 64, 128, 256]，levels=4
        ranks = [rank for i in range(levels - 1)]  # 秩为[2, 2, 2],levels=4

        self.estimator = Illumination_Estimator(n_fea_middle=dim)  # 光照信息估计

        # -- Patch Embedding
        # self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)  # [b,c,h,w] -> [b,dim,h,w]
        self.embedding = nn.Conv2d(inp_channels, self.dim, 3, 1, 1, bias=False)
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

    def forward(self, img, labels=None):

        img = self.check_image_size(img)  # 填充图像的高度和宽度，使其是 self.padder_size 的倍数。

        illu_fea, illu_map = self.estimator(img)  # 光照信息估计  # illu_fea: [b,c,h,w], illu_map: [b,c=3,h,w]
        input_img = img * illu_map + img

        feats = self.embedding(input_img)  # 对输入图像进行重叠图像块嵌入

        self.total_loss = 0  # 总损失
        enc_feats = []  # 编码器特征列表
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats, illu_fea)  # 编码器残差块组
            enc_feats.append(feats)
            feats = downsample(feats)  # 下采样层
            illu_fea = downsample(illu_fea)  # 光照信息估计下采样层

        feats = self.latent(feats, illu_fea)  # latent编码器残差块组
        freq_emb = self.freq_embed(feats)  # 将经过latent编码器的特征图进行频域嵌入

        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)  # 上采样层
            illu_fea = upsample(illu_fea)  # 光照信息估计上采样层
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))  # 特征融合层
            feats = block(feats, freq_emb)  # 解码器残差块组
            self.total_loss += block.loss  # 解码器残差块组的损失累加

        feats = self.refinement(feats, illu_fea)  # 精炼层
        out = self.output(feats) + img  # 将精炼层的特征图通过1x1卷积层调整通道数输出，并与输入图像进行残差连接，得到输出图像

        self.total_loss /= sum(self.num_dec_blocks)  # 计算总的辅助损失的平均值
        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # 需要填充的高度和宽度，以确保图像的高度和宽度是 self.padder_size 的倍数。
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)  # 使用F.pad函数填充图像。
        return x


if __name__ == '__main__':
    model = LumiSenseMoEIR(rank=2, num_blocks=[4, 6, 6, 8],
                           num_dec_blocks=[2, 4, 4],
                           levels=4,
                           dim=48,
                           num_refinement_blocks=4,
                           with_complexity=True,
                           complexity_scale="max",
                           stage_depth=[1, 1, 1],
                           depth_type="constant",
                           rank_type="spread",
                           num_experts=4,
                           topk=1,
                           expert_layer=FFTAttention).cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    _ = model(x)
    print(model.total_loss)
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery',
                                         torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2))
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))
