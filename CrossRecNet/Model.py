from functools import partial
import torch
from torch import concat, nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
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

    def forward(self, _, img, eval=False):
        # print(self.to_patch_embedding)

        if not eval:
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
        attn_out, _ = self.cross_attn(cross, dec, dec)
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
        super().__init__()
        self.patch_num = (img_size // patch_size) ** 2
        # self.onset_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim, use_cls_token=False)
        # self.flow_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim, use_cls_token=False)
        # self.onset_encoder = TransformerEncoder(embed_dim, heads, onset_depth, dim_feedforward, dropout)
        # self.flow_encoder = TransformerEncoder(embed_dim, heads, flow_depth, dim_feedforward, dropout)
        self.onset_encoder = Hencoder(img_size, patch_size, embed_dim, (img_size // patch_size) ** 2, dim_feedforward,
                                      3, heads, dropout)
        self.flow_encoder = Hencoder(img_size, patch_size, embed_dim, (img_size // patch_size) ** 2, dim_feedforward, 3,
                                     heads, dropout)
        self.decoder = InterleavedReconstruction()
        # self.projection_head = nn.Sequential(
        #     nn.LayerNorm(2 * embed_dim * self.patch_num),
        #     nn.Linear(2 * embed_dim * self.patch_num, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128)
        # )

        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(128),
        #     nn.Linear(128, num_classes)
        # )
        # self.flatten = nn.Flatten()

    def forward(self, onset_pair, flow_pair, eval=False):
        # if eval:
        #     onset_feat = self.onset_encoder(onset_pair)
        #     flow_feat = self.flow_encoder(flow_pair)
        #     flow_feat = self.flatten(flow_feat)
        #     onset_feat = self.flatten(onset_feat)
        #     feat = torch.cat((onset_feat, flow_feat), dim=1)
        #     feat = F.normalize(self.projection_head(feat), dim=1)
        #     return self.classifier(feat)

        B = onset_pair.shape[2]
        onset1, onset2 = onset_pair[:, 0], onset_pair[:, 1]
        flow1, flow2 = flow_pair[:, 0], flow_pair[:, 1]

        onset_feat1, onset_feat2 = self.onset_encoder(onset1), self.onset_encoder(onset2)

        flow_feat1, flow_feat2 = self.flow_encoder(flow1), self.flow_encoder(flow2)

        apex_reconstructed1, apex_reconstructed2 = self.decoder(flow_feat2, onset_feat1), self.decoder(flow_feat1,
                                                                                                       onset_feat2)

        # onset_feat1 = self.flatten(onset_feat1)  # [B, patch_num*embed_dim]
        # flow_feat1 = self.flatten(flow_feat1)
        # feat1 = torch.concat((onset_feat1, flow_feat1), dim=-1)
        # onset_feat2 = self.flatten(onset_feat2)
        # flow_feat2 = self.flatten(flow_feat2)
        # feat2 = torch.concat((onset_feat2, flow_feat2), dim=-1)

        # feat1 = F.normalize(self.projection_head(feat1), dim=1)
        # feat2 = F.normalize(self.projection_head(feat2), dim=1)

        # # feat1 = self.flatten(torch.cat((onset_feat1, flow_feat1), dim=1))
        # # feat2 = self.flatten(torch.cat((onset_feat2, flow_feat2), dim=1))
        # classification1 = self.classifier(feat1)
        # classification2 = self.classifier(feat2)

        return (apex_reconstructed1, apex_reconstructed2), (flow_feat1, flow_feat2)


class Vit(nn.Module):
    def __init__(self, num_classes=3, embed_dim=512, img_size=224, patch_size=14, onset_depth=6, flow_depth=6,
                 dec_depth=6, heads=8, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        self.onset_encoder = Hencoder(img_size, patch_size, embed_dim, (img_size // patch_size) ** 2, dim_feedforward,
                                      3, heads, dropout)
        self.flow_encoder = Hencoder(img_size, patch_size, embed_dim, (img_size // patch_size) ** 2, dim_feedforward, 3,
                                     heads, dropout)
        # self.decoder = TransformerDecoder(embed_dim, (img_size, img_size), patch_size, heads, dec_depth, dim_feedforward)
        self.patch_num = (img_size // patch_size) ** 2
        self.classifier = self.classifier = nn.Sequential(
            nn.LayerNorm(512 * 4),
            nn.Linear(512 * 4, num_classes)
        )
        self.flatten = nn.Flatten()

    def forward(self, onset_pair, flow_pair, eval=False):
        if eval:
            onset_feat = self.onset_encoder(onset_pair)
            flow_feat = self.flow_encoder(flow_pair)
            flow_feat = self.flatten(flow_feat)
            onset_feat = self.flatten(onset_feat)
            flow_feat = torch.cat((onset_feat, flow_feat), dim=1)
            return self.classifier(flow_feat)
        flow1 = flow_pair[:, 0]
        onset = onset_pair[:, 0]
        onset_feat = self.onset_encoder(onset)
        flow_feat = self.flow_encoder(flow1)
        flow_feat = self.flatten(flow_feat)
        onset_feat = self.flatten(onset_feat)
        flow_feat = torch.cat((onset_feat, flow_feat), dim=1)
        # print(flow_feat.shape)
        flow_feat = self.classifier(flow_feat)

        return flow_feat


class Hencoder(nn.Module):
    def __init__(self, img_size, patch_size, dim, seq_len, ffn_dim, channel=3, heads=8, dropout=0.1):
        super().__init__()

        # 下采样卷积层
        self.downsample_conv = nn.Sequential(
            nn.Conv2d(channel, dim // 2, kernel_size=3, stride=2, padding=1),  # 第一个卷积层，减小尺寸
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),  # 第二个卷积层，进一步减小尺寸
            nn.ReLU()
        )

        # 将下采样后的特征图转换为 Transformer 输入格式
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(dim * (patch_size ** 2), dim, kernel_size=1)  # 投影到嵌入维度
        )

        # Transformer 编码器列表
        # layer_num = [2, 2, 8]
        layer_num = [2, 2, 2]
        self.enc_list = nn.ModuleList([])
        for layer in range(3):
            self.enc_list.append(
                nn.ModuleList([
                    Transformer(dim=dim,
                                seq_len=seq_len,
                                depth=layer_num[layer],
                                heads=heads,
                                mlp_mult=ffn_dim,
                                dropout=dropout),
                    Aggregate(dim, dim)
                ])
            )

        # 其他参数
        self.patch_num = img_size // (patch_size * 4)  # 因为下采样了两次（stride=2），所以要除以4
        self.hierarchical = [2 ** i for i in range(3)]
        self.hierarchical.reverse()
        # self.total_loss = 0

    def forward(self, x):

        # self.total_loss = 0
        # [B, C, H, W] -> [B, D, H', W'] 通过下采样卷积层
        x = self.downsample_conv(x)

        # [B, D, H', W'] -> [B, D, H'', W''] 转换为 patch embedding
        x = self.to_patch_embedding(x)
        b, c, h, w = x.shape

        # Transformer 编码器处理
        for layer_num, (enc, down_samp) in enumerate(self.enc_list):
            x = rearrange(x, 'b d (h p1) (w p2) -> (b p1 p2) d h w', p1=self.hierarchical[layer_num],
                          p2=self.hierarchical[layer_num])
            x = enc(x)
            x = rearrange(x, '(b p1 p2) d h w -> b d (h p1) (w p2)', p1=self.hierarchical[layer_num],
                          p2=self.hierarchical[layer_num])
            # self.total_loss += enc.loss
            if not layer_num == 2:
                x = down_samp(x)

        # 最终输出形状调整为 [B, SeqLen, Dim]
        x = rearrange(x, 'b d h w -> b (h w) d')
        # self.total_loss /= len(self.enc_list)  # 计算平均损失
        return x


class InterleavedReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始投影（保持256维）
        self.proj_flow = nn.Linear(256, 256)
        self.proj_face = nn.Linear(256, 256)

        # 第一级处理 (2x2 -> 4x4)
        self.attn1 = CrossAttentionBlock(256)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU()
        )

        # 第二级处理 (4x4 -> 8x8)
        self.attn2 = CrossAttentionBlock(256)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU()
        )

        # 第三级处理 (8x8 -> 16x16)
        self.attn3 = CrossAttentionBlock(256)
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU()
        )

        # 最终重建
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, flow_feat, face_feat):
        B = flow_feat.shape[0]

        # --- Level 1 ---
        flow = self.proj_flow(flow_feat)  # [B,4,256]
        face = self.proj_face(face_feat)  # [B,4,256]
        flow, face = self.attn1(flow, face)
        fused = (flow + face) / 2
        x = fused.transpose(1, 2).view(B, 256, 2, 2)  # [B,256,2,2]
        x = self.upsample1(x)  # [B,256,4,4]

        # --- Level 2 ---
        # 残差分支处理
        flow = flow.transpose(1, 2).view(B, 256, 2, 2)  # [B,256,2,2]
        flow = F.interpolate(flow, scale_factor=2)  # [B,256,4,4]
        flow = flow.flatten(2).transpose(1, 2)  # [B,16,256]
        flow = flow.view(B, 4, 4, 256).mean(dim=2)  # [B,4,256] (序列长度调整)

        face = face.transpose(1, 2).view(B, 256, 2, 2)
        face = F.interpolate(face, scale_factor=2)
        face = face.flatten(2).transpose(1, 2)
        face = face.view(B, 4, 4, 256).mean(dim=2)  # [B,4,256]

        # 交叉注意力 + 残差连接
        flow, face = self.attn2(flow, face)  # [B,4,256]
        fused = (flow + face) / 2
        fused_spatial = fused.transpose(1, 2).view(B, 256, 2, 2)  # [B,256,2,2]
        fused_spatial = F.interpolate(fused_spatial, scale_factor=2)  # [B,256,4,4]
        x = x + fused_spatial  # [B,256,4,4]
        x = self.upsample2(x)  # [B,256,8,8]

        # --- Level 3 --- 
        # (类似Level 2的处理，略)

        # 最终输出
        x = F.interpolate(x, size=(112, 112), mode='bilinear')  # [B,256,112,112]
        return self.final_conv(x)  # [B,3,112,112]


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.flow_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.face_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, flow, face):
        # flow/face: B*N*C (N=4/16/64...)
        flow2 = self.norm1(flow + self.flow_attn(flow, face, face)[0])
        face2 = self.norm2(face + self.face_attn(face, flow, flow)[0])
        return flow2, face2


if __name__ == '__main__':
    model = Hencoder(img_size=224, patch_size=14, dim=256, seq_len=16, ffn_dim=1024, channel=3, heads=8,
                     dropout=0.1).cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    _ = model(x)
    # print(model.total_loss)
    # Memory usage
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery',
                                         torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2))

    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops))
