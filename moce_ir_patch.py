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

def patch_split(x, patch_size):
    """将特征图分割为块"""
    b, c, h, w = x.shape
    x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=patch_size, p2=patch_size)
    return x, (h//patch_size, w//patch_size)

def patch_merge(x, original_shape, patch_size):
    """将块重组为特征图"""
    h_patches, w_patches = original_shape
    x = rearrange(x, 'b (h w) c p1 p2 -> b c (h p1) (w p2)',
                  h=h_patches, w=w_patches, p1=patch_size, p2=patch_size)
    return x

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


class PatchedSparseDispatcher(object):
    def __init__(self, num_experts, gates, num_patches):
        """支持展平后的2D gates输入: [B*num_patches, num_experts]"""
        self._gates = gates  # [B*num_patches, num_experts]
        self._num_experts = num_experts
        self._num_patches = num_patches

        # 获取非零元素的全局索引
        nonzero_indices = torch.nonzero(gates)  # [K,2]
        self._flat_indices = nonzero_indices[:, 0]  # 展平后的样本索引
        self._expert_index = nonzero_indices[:, 1]  # 专家索引

        # 恢复原始batch和块索引
        batch_patch_idx = np.unravel_index(
            self._flat_indices.cpu().numpy(),
            (gates.size(0) // num_patches, num_patches))
        self._batch_index = torch.tensor(batch_patch_idx[0], device=gates.device)
        self._patch_index = torch.tensor(batch_patch_idx[1], device=gates.device)

        # 统计每个专家的样本数
        self._part_sizes = (gates > 0).sum(0).tolist()
        self._nonzero_gates = gates[self._flat_indices, self._expert_index]

    def dispatch(self, inp):
        """分发展平的输入: [B*num_patches, ...] -> 专家列表"""
        selected = inp[self._flat_indices]
        return torch.split(selected, self._part_sizes, dim=0)

    def combine(self, expert_out, original_batch_size):
        """合并专家输出并恢复形状"""
        # 拼接所有专家输出 [total_selected, ...]
        stitched = torch.cat(expert_out, dim=0)

        # 初始化全零张量 [B*num_patches, ...]
        combined_flat = torch.zeros(
            (self._gates.size[0], * stitched.shape[1:]),
            device=stitched.device
        )

        # 加权求和
        if self._nonzero_gates is not None:
            stitched = stitched * self._nonzero_gates.view(-1, 1, 1, 1)

        # 使用index_add累加结果
        combined_flat.index_add_(
            0, self._flat_indices, stitched
        )

        # 恢复原始形状 [B, num_patches, ...]
        return combined_flat.view(
            original_batch_size, self._num_patches, *combined_flat.shape[1:]
        )
##########################################################################
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


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
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
class PatchAdapterLayer(nn.Module):
    def __init__(self, dim, rank, num_experts=4, top_k=2,
                 expert_layer=FFTAttention,stage_depth=1, depth_type="lin",
                 rank_type="constant", freq_dim=128,
                 with_complexity=False, complexity_scale="min", patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.num_experts = num_experts
        self.top_k = top_k



        # 专家初始化（与原实现相同）
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
        expert_complexity = torch.tensor(
            [sum(p.numel() for p in expert.parameters()) for expert in self.experts])  # 计算专家复杂度
        # 修改路由模块
        self.routing = PatchRoutingFunction(
            dim, freq_dim, num_experts, top_k,
            complexity=expert_complexity,
            patch_size=patch_size,
            use_complexity_bias=with_complexity,
            complexity_scale=complexity_scale
        )

    def forward(self, x, freq_emb, shared):
        B, C, H, W = x.shape
        patch_size = self.patch_size
        h_patches = H // patch_size
        w_patches = W // patch_size
        num_patches = h_patches * w_patches

        # 分割输入为块并展平 [B*num_patches, C, p, p]
        x_patches = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size)
        shared_patches = rearrange(shared, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size)

        # 路由计算 (注意输入是展平后的x_patches)
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x_patches, freq_emb)
        self.loss = aux_loss

        if self.training:
            # 使用改进的分发器
            dispatcher = PatchedSparseDispatcher(
                self.num_experts,
                gates,
                num_patches=num_patches
            )

            # 分发输入
            expert_inputs = dispatcher.dispatch(x_patches)
            expert_shared_inputs = dispatcher.dispatch(shared_patches)

            # 专家计算
            expert_outputs = [
                self.experts[exp](expert_inputs[exp], expert_shared_inputs[exp])
                for exp in range(self.num_experts)
            ]

            # 合并输出 [B, num_patches, C, p, p]
            combined = dispatcher.combine(expert_outputs, original_batch_size=B)

            # 恢复空间维度 [B, C, H, W]
            out = rearrange(combined, 'b (h w) c p1 p2 -> b c (h p1) (w p2)',
                            h=h_patches, w=w_patches, p1=patch_size, p2=patch_size)
        else:
            # 推理优化路径
            selected_experts = [self.experts[i] for i in top_k_indices[0]]  # 取第一个样本的专家选择
            expert_outputs = []

            # 按批次处理
            for b in range(B):
                # 获取当前批次的块 [num_patches, C, p, p]
                batch_patches = x_patches[b * num_patches:(b + 1) * num_patches]
                batch_shared = shared_patches[b * num_patches:(b + 1) * num_patches]

                # 选择专家并计算
                expert_out = torch.stack([
                    expert(batch_patches, batch_shared)
                    for expert in selected_experts
                ], dim=1)  # [num_patches, K, C, p, p]

                # 加权融合
                weights = top_k_values[b].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                weighted_out = (expert_out * weights).sum(dim=1)  # [num_patches, C, p, p]
                expert_outputs.append(weighted_out)

            # 合并批次 [B*num_patches, C, p, p]
            out_flat = torch.cat(expert_outputs, dim=0)
            out = rearrange(out_flat, '(b h w) c p1 p2 -> b c (h p1) (w p2)',
                            b=B, h=h_patches, w=w_patches, p1=patch_size, p2=patch_size)

        return self.proj_out(out)


class PatchRoutingFunction(nn.Module):
    def __init__(self, dim, freq_dim, num_experts, k, complexity,
                 patch_size=16, use_complexity_bias=True, complexity_scale="max"):
        super().__init__()
        self.patch_size = patch_size
        self.gate = nn.Sequential(
            nn.Conv2d(dim, num_experts, kernel_size=patch_size, stride=patch_size),  # 块级特征提取
            Rearrange('b e h w -> b (h w) e')
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
        # 输入x: [B, C, H, W]
        batch_size = x.shape[0]

        # 块级路由门控
        spatial_gates = self.gate(x)  # [B, num_patches, num_experts]
        freq_gates = self.freq_gate(freq_emb).unsqueeze(1)  # [B, 1, num_experts]
        logits = spatial_gates + freq_gates

        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))

        # 添加噪声与Top-K选择
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        # 计算辅助损失
        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0

        # 生成块级路由掩码
        gates = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values, aux_loss

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=(0, 1))  # 跨batch和块求和
        importance = importance * (self.complexity * self.tau) if self.use_complexity_bias else importance
        imp_mean = importance.mean()
        imp_std = importance.std()
        return (imp_std / (imp_mean + 1e-8)) ** 2

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
## Encoder Block
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
        


##########################################################################
## Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, expert_layer, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, patch_size=16, depth_type=None, rank_type=None, stage_depth=None, freq_dim:int=128, with_complexity: bool=False):
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
        
        # self.adapter = AdapterLayer(
        #     dim, rank,
        #     top_k=top_k, num_experts=num_experts, expert_layer=expert_layer, freq_dim=freq_dim,
        #     depth_type=depth_type, rank_type=rank_type, stage_depth=stage_depth,
        #     with_complexity=with_complexity, complexity_scale=complexity_scale
        # )
        # 修改Adapter初始化
        self.adapter = PatchAdapterLayer(
            dim, rank,
            top_k=top_k, num_experts=num_experts,
            expert_layer=expert_layer, freq_dim=freq_dim,
            depth_type=depth_type, rank_type=rank_type, stage_depth=stage_depth,
            with_complexity=with_complexity, complexity_scale=complexity_scale,
            patch_size=patch_size,  # 新增参数
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
                # DecoderBlock(
                #     dim, num_heads, ffn_expansion, bias, LayerNorm_type,
                #     expert_layer=expert_layer, rank=rank, num_experts=num_experts, top_k=top_k,
                #     stage_depth=stage_depth, freq_dim=freq_dim, complexity_scale=complexity_scale,
                #     depth_type=depth_type, rank_type=rank_type, with_complexity=with_complexity
                # )
                DecoderBlock(
                    dim, num_heads, ffn_expansion, bias, LayerNorm_type,
                    expert_layer=expert_layer, rank=rank, num_experts=num_experts, top_k=top_k,
                    stage_depth=stage_depth, freq_dim=freq_dim, complexity_scale=complexity_scale,
                    depth_type=depth_type, rank_type=rank_type, with_complexity=with_complexity,
                    patch_size=16,  # 新增参数
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

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
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
                ffn_expansion_factor = 2,
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
        
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        
        dims = [dim*2**i for i in range(levels)]
        ranks = [rank for i in range(levels-1)]

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        self.freq_embed = FrequencyEmbedding(dims[-1])
                
        # -- Encoder --        
        self.enc = nn.ModuleList([])
        for i in range(levels-1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i], 
                    num_blocks=num_blocks[i], 
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=True,),
                Downsample(dim*2**i)
                ])
            )
        
        # -- Latent --
        self.latent = EncoderResidualGroup(
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
    # 256 -> 128 -> 64 -> 32
    # test
    model = MoCEIR(rank=2, num_blocks=[4,6,6,8], num_dec_blocks=[2,4,4], levels=4, dim=48, num_refinement_blocks=4, 
                   with_complexity=True, complexity_scale="max", stage_depth=[1,1,1], depth_type="constant", rank_type="spread", 
                   num_experts=4, topk=1, expert_layer=FFTAttention).cuda()

    x = torch.randn(1, 3, 256, 256).cuda()
    _ = model(x)
    print(model.total_loss)
    # Memory usage  
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))
  
    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))