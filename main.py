import torch
from torch import nn


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
model = TransformerDecoder()
print(model)