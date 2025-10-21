import numbers
from typing import List

from torch import nn
from einops import rearrange
import torch
from torch.nn import functional as F


class Illumination_Estimator(nn.Module):
    """
    用于估计光照信息的网络结构
    """

    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
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

    def forward(self,x_in, illu_fea):
        """

        :param x_in: [b,c,h,w]
        :param illu_fea: [b,c,h,w]
        :return: [b,c,h,w]
        """
        b, c, h, w = x_in.shape  # [1, 48, 224, 224]
        qkv = self.qkv_dwconv(self.qkv(x_in))  # [1, 144, 224, 224]
        q,k,v = qkv.chunk(3, dim=1)  # [1, 48, 224, 224]

        # 重排 QKV 以适应多头注意力机制
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [1, 1, 48, 50176]
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [1, 1, 48, 50176]
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [1, 1, 48, 50176]
        illu_attn =illu_fea

        illu_attn = rearrange(illu_attn, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        v = v * illu_attn
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


## Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type)
        ])

        # self.mixer = Attention(dim, num_heads, bias)
        self.mixer = IG_MSA(dim=dim, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x,illu_fea):
        x = self.norms[0](x)
        x = x + self.mixer(x,illu_fea)
        x = self.norms[1](x)
        x = x + self.ffn(x)
        return x

        # x = x + self.mixer(self.norms[0](x))
        # x = x + self.ffn(self.norms[1](x))


class EncoderResidualGroup(nn.Module):
    def __init__(self,
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type))

    def forward(self, x,illu_fea):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x,illu_fea)
            i += 1
        return x


## Overlapped image patch embedding with 3x3 Conv
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

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Model(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=32,
                 levels=4,
                 heads=[1,1,1,1],
                 num_blocks=[1,1,1,3],
                 ffn_expansion_factor=2,
                 LayerNorm_type='WithBias',  # 'WithBias' or 'BiasFree'
                 bias=False,
                 ):
        super(Model, self).__init__()

        self.levels = levels
        self.num_blocks = num_blocks

        self.estimator = Illumination_Estimator(n_fea_middle=dim)

        dims = [dim * 2 ** i for i in range(levels)]

        self.patch_embed = OverlapPatchEmbed(in_c=in_channels, embed_dim=dim, bias=False)


        # Encoder
        self.enc = nn.ModuleList([])
        for i in range(levels-1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i],
                    num_blocks=num_blocks[i],
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type,
                    bias=True,),
                Downsample(dim*2**i)
              ])
            )
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1],
            num_heads=heads[-1],
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True, )
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)  # 卷积输出层
        self.total_loss = None

    def forward(self,x):

        # 光照估计
        illu_fea, _ = self.estimator(x)
        # 光照图归一化
        feats = self.patch_embed(x)

        self.total_loss = 0
        enc_feats = []

        for i,(block, downsample) in enumerate(self.enc):
            feats = block(feats,illu_fea)
            enc_feats.append(feats)
            feats = downsample(feats)
        feats = self.latent(feats, illu_fea)
        return feats, enc_feats




if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224).cuda()
    model = Model(dim=48, levels=4, heads=[1,1,1,1], num_blocks=[4,6,6,8]).cuda()
    out,_ = model(img)
    print(out.shape)
