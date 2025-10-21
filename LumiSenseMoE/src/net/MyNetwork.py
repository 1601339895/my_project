# -*- coding: utf-8 -*-
# File  : MyNetwork.py
# Author: HeLei
# Date  : 2025/4/17
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import nn
from LumiSenseMoE.src.net.basenet import HighPassConv2d
from LumiSenseMoE.src.net.Encoder import EncoderResidualGroup
from LumiSenseMoE.src.net.Decoder import DecoderResidualGroup,FFTAttention


##########################################################################
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

        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
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
        super(MoCEIR, self).__init__()

        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks

        dims = [dim * 2 ** i for i in range(levels)]
        ranks = [rank for i in range(levels - 1)]

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        self.freq_embed = FrequencyEmbedding(dims[-1])

        # -- Encoder --
        self.enc = nn.ModuleList([])
        for i in range(levels - 1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i],
                    num_blocks=num_blocks[i],
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type, bias=True ),
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

        # -- Refinement --
        heads = heads[::-1]
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks,
            num_heads=heads[0],
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True, )

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.total_loss = None

    def forward(self, x, labels=None):

        feats = self.patch_embed(x)

        self.total_loss = 0
        enc_feats = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)

        feats = self.latent(feats)
        freq_emb = self.freq_embed(feats)

        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            feats = block(feats, freq_emb)
            self.total_loss += block.loss

        feats = self.refinement(feats)
        x = self.output(feats) + x

        self.total_loss /= sum(self.num_dec_blocks)
        return x

if __name__ == "__main__":
    # test
    model = MoCEIR(rank=2, num_blocks=[4, 6, 6, 8], num_dec_blocks=[2, 4, 4], levels=4, dim=48, num_refinement_blocks=4,
                   with_complexity=True, complexity_scale="max", stage_depth=[1, 1, 1], depth_type="constant",
                   rank_type="spread",
                   num_experts=4, topk=1, expert_layer=FFTAttention).cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    _ = model(x)
    print(model.total_loss)
    # Memory usage
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery',
                                         torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2))

    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))

