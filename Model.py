from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.swin_transformer import swin_tiny_patch4_window7_224, swin_base_patch4_window7_224


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )


class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h=h, w=w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class HTNet(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            heads,
            num_hierarchies,
            block_repeats,
            mlp_mult=4,
            channels=3,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2  #
        fmap_size = image_size // patch_size  #
        blocks = 2 ** (num_hierarchies - 1)  #

        seq_len = (fmap_size // blocks) ** 2  # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img, training=True):
        # print(self.to_patch_embedding)

        if training:
            x = self.to_patch_embedding(img[:, 0])
        else:
            x = self.to_patch_embedding(img)
        # print(x.shape)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)
        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            # print(level,":",x.shape)
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x = transformer(x)
            # print(level,":",x.shape)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x = aggregate(x)
            # print(level,":",x.shape)
        return self.mlp_head(x)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_channels=3, embed_dim=512, use_cls_token=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, (img_size // patch_size) ** 2 + (1 if use_cls_token else 0), embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # 添加 CLS token
        x = x + self.pos_embed
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                       dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec, cross):
        attn_out, _ = self.self_attn(dec, dec, dec)
        dec = self.norm1(dec + self.dropout(attn_out))
        attn_out, _ = self.cross_attn(cross, cross, dec)
        dec = self.norm1(dec + self.dropout(attn_out))

        # Feed-Forward Network + Residual + Norm
        ffn_out = self.ffn(dec)
        dec = self.norm2(dec + self.dropout(ffn_out))

        return dec


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, img_size=(224, 224), patch_size=14, num_heads=8, depth=6, ffn_dim=1024):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(depth)
        ])
        self.patch_size = patch_size
        self.upconv = nn.ConvTranspose2d(embed_dim, 3, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        self.img_size = img_size

    def forward(self, onset_feat, flow_feat):
        for layer in self.layers:
            onset_feat = layer(onset_feat, flow_feat)

        B, N, C = onset_feat.shape
        H, W = self.img_size
        onset_feat = onset_feat.permute(0, 2, 1).view(B, C, H // self.patch_size, W // self.patch_size)
        x = self.upconv(onset_feat)
        return x


class MicroExpressionNet(nn.Module):
    def __init__(self, num_classes=3, embed_dim=512, img_size=224, patch_size=14, onset_depth=6, flow_depth=6,
                 dec_depth=6, heads=8, dim_feedforward=1024, dropout=0.2):
        """

        :param num_classes: 类别数
        :param embed_dim:  embedding 维度
        :param img_size:  图像大小
        :param patch_size: patch 大小
        :param onset_depth: onset编码器层数
        :param flow_depth:  flow编码器层数
        :param dec_depth:   解码器层数
        :param heads:      多头注意力层数
        :param dim_feedforward:       前馈网络层数
        :param dropout:     dropout 率
        """
        super().__init__()
        self.patch_num = (img_size // patch_size) ** 2  # 计算patch数量
        self.onset_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim, use_cls_token=False)  # 定义onset嵌入
        self.flow_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim, use_cls_token=False)  # 定义flow嵌入
        self.onset_encoder = TransformerEncoder(embed_dim, heads, onset_depth, dim_feedforward, dropout)  # 定义onset编码器
        self.flow_encoder = TransformerEncoder(embed_dim, heads, flow_depth, dim_feedforward, dropout)  # 定义flow编码器
        self.decoder = TransformerDecoder(embed_dim, (img_size, img_size), patch_size, heads, dec_depth,
                                          dim_feedforward)  # 定义解码器
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * embed_dim * self.patch_num),
            nn.Linear(2 * embed_dim * self.patch_num, num_classes)
        )  # 分类器
        self.flatten = nn.Flatten()

    def forward(self, onset_pair, flow_pair, eval=False):
        if eval:
            onset_feat = self.onset_encoder(self.onset_embed(onset_pair))
            flow_feat = self.flow_encoder(self.flow_embed(flow_pair))
            flow_feat = self.flatten(flow_feat)
            onset_feat = self.flatten(onset_feat)
            flow_feat = torch.cat((onset_feat, flow_feat), dim=1)
            return self.classifier(flow_feat)

        B = onset_pair.shape[2]
        onset1, onset2 = onset_pair[:, 0], onset_pair[:, 1]
        flow1, flow2 = flow_pair[:, 0], flow_pair[:, 1]

        onset_feat1, onset_feat2 = self.onset_encoder(self.onset_embed(onset1)), self.onset_encoder(
            self.onset_embed(onset2))

        flow_feat1, flow_feat2 = self.flow_encoder(self.flow_embed(flow1)), self.flow_encoder(self.flow_embed(flow2))

        apex_reconstructed1, apex_reconstructed2 = self.decoder(onset_feat1, flow_feat2), self.decoder(onset_feat2,
                                                                                                       flow_feat1)

        feat1 = self.flatten(torch.cat((onset_feat1, flow_feat1), dim=1))
        feat2 = self.flatten(torch.cat((onset_feat2, flow_feat2), dim=1))
        classification1 = self.classifier(feat1)
        classification2 = self.classifier(feat2)

        return (classification1, classification2), (apex_reconstructed1, apex_reconstructed2)


class Vit(nn.Module):
    def __init__(self, num_classes=3, embed_dim=512, img_size=224, patch_size=14, onset_depth=6, flow_depth=6,
                 dec_depth=6, heads=8, dim_feedforward=1024, dropout=0.2):
        """

        :param num_classes:
        :param embed_dim:
        :param img_size:
        :param patch_size:
        :param onset_depth:
        :param flow_depth:
        :param dec_depth:
        :param heads:
        :param dim_feedforward:
        :param dropout:
        """
        super().__init__()
        self.onset_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim, use_cls_token=False)
        self.flow_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim, use_cls_token=False)
        self.onset_encoder = TransformerEncoder(embed_dim, heads, onset_depth, dim_feedforward, dropout)
        self.flow_encoder = TransformerEncoder(embed_dim, heads, onset_depth, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(embed_dim, (img_size, img_size), heads, dec_depth, dim_feedforward)
        self.patch_num = (img_size // patch_size) ** 2
        self.classifier = self.classifier = nn.Sequential(
            nn.LayerNorm(2 * embed_dim * self.patch_num),
            nn.Linear(2 * embed_dim * self.patch_num, num_classes)
        )
        self.flatten = nn.Flatten()

    def forward(self, onset_pair, flow_pair, training=True):
        if not training:
            onset_feat = self.onset_encoder(self.onset_embed(onset_pair))
            flow_feat = self.flow_encoder(self.flow_embed(flow_pair))
            flow_feat = self.flatten(flow_feat)
            onset_feat = self.flatten(onset_feat)
            flow_feat = torch.cat((onset_feat, flow_feat), dim=1)
            return self.classifier(flow_feat)

        flow1 = flow_pair[:, 0]
        onset = onset_pair[:, 0]
        onset_feat = self.onset_encoder(self.onset_embed(onset))
        flow_feat = self.flow_encoder(self.flow_embed(flow1))
        flow_feat = self.flatten(flow_feat)
        onset_feat = self.flatten(onset_feat)
        flow_feat = torch.cat((onset_feat, flow_feat), dim=1)
        flow_feat = self.classifier(flow_feat)
        # print(flow_feat.shape)
        # classification1 = self.classifier(flow_feat)
        return flow_feat
