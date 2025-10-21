import math

import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple
import torch
import torch.fft
import torch.nn.functional as F
from torch.distributions.normal import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


class HighFreqLocal(nn.Module):
    """高频局部专家 - 使用小卷积核捕捉局部高频细节"""

    def __init__(self, rank, kernel_size, patch_size):
        super().__init__()
        # 使用小卷积核(3x3)捕捉局部细节
        self.conv = nn.Sequential(
            nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank),
            nn.GELU(),
            nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank)
        )

    def forward(self, x):
        return self.conv(x)


class LowFreqLocal(nn.Module):
    """低频局部专家 - 使用大卷积核捕捉局部低频信息"""

    def __init__(self, rank, kernel_size, patch_size):
        super().__init__()
        # 使用大卷积核(7x7)捕捉更大范围的低频信息
        padding = kernel_size // 2 if kernel_size else 3
        self.conv = nn.Sequential(
            nn.Conv2d(rank, rank, kernel_size=kernel_size, padding=padding, groups=rank),
            nn.GELU(),
            nn.AvgPool2d(3, stride=1, padding=1)  # 平滑操作增强低频
        )

    def forward(self, x):
        return self.conv(x)


class HighFreqGlobal(nn.Module):
    """高频全局专家 - 在频域增强高频分量"""

    def __init__(self, rank, kernel_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # 可学习的高通滤波器参数
        self.high_pass_gain = nn.Parameter(torch.tensor(2.0))
        self.low_pass_decay = nn.Parameter(torch.tensor(0.5))

    def fft_shift(self, x):
        """将零频移到频谱中心"""
        return torch.fft.fftshift(x, dim=(-2, -1))

    def ifft_shift(self, x):
        """将零频移回原位置"""
        return torch.fft.ifftshift(x, dim=(-2, -1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 傅里叶变换到频域
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_fft = torch.view_as_complex(x_fft)

        # 分离幅度和相位
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # 创建高频增强滤波器
        freqs_y = torch.fft.fftfreq(H, device=x.device)[:, None]
        freqs_x = torch.fft.rfftfreq(W, device=x.device)[None, :]
        freq_grid = torch.sqrt(freqs_y ** 2 + freqs_x ** 2)

        # 高通滤波器：增强高频，衰减低频
        high_pass = (1 - torch.exp(-self.high_pass_gain * freq_grid)) * \
                    torch.exp(-self.low_pass_decay * freq_grid)
        high_pass = high_pass.clamp(0, 3)  # 限制增益范围

        # 应用高通滤波器
        magnitude = magnitude * high_pass

        # 重建复数频谱
        x_fft = torch.polar(magnitude, phase)

        # 逆傅里叶变换回空域
        x_filtered = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        return x_filtered


class LowFreqGlobal(nn.Module):
    """低频全局专家 - 在频域保留低频分量"""

    def __init__(self, rank, kernel_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # 可学习的低通滤波器参数
        self.low_pass_gain = nn.Parameter(torch.tensor(1.0))
        self.high_pass_decay = nn.Parameter(torch.tensor(1.0))

    def fft_shift(self, x):
        """将零频移到频谱中心"""
        return torch.fft.fftshift(x, dim=(-2, -1))

    def ifft_shift(self, x):
        """将零频移回原位置"""
        return torch.fft.ifftshift(x, dim=(-2, -1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 傅里叶变换到频域
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_fft = torch.view_as_complex(x_fft)

        # 分离幅度和相位
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # 创建低通滤波器
        freqs_y = torch.fft.fftfreq(H, device=x.device)[:, None]
        freqs_x = torch.fft.rfftfreq(W, device=x.device)[None, :]
        freq_grid = torch.sqrt(freqs_y ** 2 + freqs_x ** 2)

        # 低通滤波器：保留低频，衰减高频
        low_pass = torch.exp(-self.high_pass_decay * freq_grid) * \
                   (1 - torch.exp(-self.low_pass_gain * freq_grid))
        low_pass = low_pass.clamp(0, 1)  # 限制增益范围

        # 应用低通滤波器
        magnitude = magnitude * low_pass

        # 重建复数频谱
        x_fft = torch.polar(magnitude, phase)

        # 逆傅里叶变换回空域
        x_filtered = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        return x_filtered


class ModExpert(nn.Module):
    """专家适配器模块"""

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


class New_AdapterLayer(nn.Module):
    def __init__(self,
                 dim: int, rank: int, num_experts: int = 4, top_k: int = 2,
                 # expert_layer: nn.Module = FFTAttention,
                 stage_depth: int = 1,
                 depth_type: str = "lin", rank_type: str = "constant",
                 freq_dim: int = 128,):
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

        # 定义四种专家类型
        expert_types = [HighFreqLocal, HighFreqGlobal, LowFreqLocal, LowFreqGlobal]

        self.experts = nn.ModuleList([
            MySequential(
                *[ModExpert(
                    dim,
                    rank=rank,
                    func=expert_types[idx],  # 根据索引选择专家类型
                    depth=depth,
                    patch_size=patch,
                    kernel_size=kernel
                ) for _ in range(1)]  # 保持原始深度结构
            )
            for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
        ])
        # self.experts = nn.ModuleList([
        #     MySequential(
        #         *[ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel)])
        #     for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
        # ])

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        # expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, freq_dim,
            num_experts=num_experts, k=top_k)

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




class FrequencyAwareRouter(nn.Module):
    def __init__(self, num_experts=4, freq_bins=8):
        super().__init__()
        self.freq_bins = freq_bins
        self.router = nn.Sequential(
            nn.Linear(freq_bins, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

    def get_frequency_energy(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        fft = torch.fft.fft2(x, norm='forward')
        magnitude = torch.abs(fft)

        # 转换为极坐标频率分布（中心为低频，外圈为高频）
        freq_grid = self._get_freq_grid(H, W).to(x.device)
        freq_dist = freq_grid.norm(dim=-1)  # 归一化频率距离 [H, W]

        # 将图像划分为若干频率区间，统计每个区间的平均能量
        bin_edges = torch.linspace(0, freq_dist.max(), self.freq_bins + 1).to(x.device)
        energy_bins = []
        for i in range(self.freq_bins):
            mask = (freq_dist >= bin_edges[i]) & (freq_dist < bin_edges[i+1])
            energy = (magnitude * mask[None, None]).sum(dim=(-1, -2))  # [B, C]
            energy_bins.append(energy.mean(dim=1))  # [B]

        energy_profile = torch.stack(energy_bins, dim=1)  # [B, freq_bins]
        return energy_profile

    def _get_freq_grid(self, H, W):
        y = torch.arange(-H//2, H//2)
        x = torch.arange(-W//2, W//2)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1)
        return grid.float()

    def forward(self, x, k=2):
        with torch.no_grad():
            energy_profile = self.get_frequency_energy(x)  # [B, freq_bins]

        logits = self.router(energy_profile)  # [B, num_experts]
        topk_weights, topk_indices = torch.topk(logits, k=k, dim=-1)
        return topk_indices, F.softmax(topk_weights, dim=-1)


class RoutingFunction(nn.Module):
    def __init__(self, dim, freq_dim, num_experts, k,
                 freq_bins=8,  # 新增频域 bins 数量
                 use_load_balance: bool = True,
                 use_importance: bool = True):
        super(RoutingFunction, self).__init__()

        # 主要图像特征 gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        )

        # 频域特征 gate
        self.freq_gate = nn.Sequential(
            nn.Linear(freq_bins, freq_dim),
            nn.ReLU(),
            nn.Linear(freq_dim, num_experts, bias=False)
        )

        # 频域设置
        self.freq_bins = freq_bins
        self.num_experts = num_experts
        self.k = k
        self.tau = 1
        self.noise_std = (1.0 / num_experts) * 1.0

        # 控制是否启用损失项
        self.use_load_balance = use_load_balance
        self.use_importance = use_importance

    def forward(self, x, freq_emb=None):
        # 提取频域特征（如果没有传入 freq_emb）  # 最好从原始图像进行频域嵌入。
        if freq_emb is None:
            freq_emb = self.get_frequency_energy(x)  # [B, freq_bins]

        # 得到两个 logits 分支
        img_logits = self.gate(x)   # [B,num_experts]        # [B, K]
        freq_logits = self.freq_gate(freq_emb)  # [B, K]
        logits = img_logits + freq_logits  # 可替换为 concat 后接 linear

        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))
            loss_load = self.load_loss(logits, logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0

        # 加噪声以增强探索
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)

        # 选择 top-k 专家
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        # 构建 gates 稀疏矩阵
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)

        return gates, top_k_indices, top_k_values, aux_loss


    def get_frequency_energy(self, x):
        B, C, H, W = x.shape
        fft = torch.fft.fft2(x, norm='forward')
        magnitude = torch.abs(fft)

        # 构建频率坐标网格
        freq_grid = self._get_freq_grid(H, W).to(x.device)
        freq_dist = freq_grid.norm(dim=-1)  # [H, W]

        # 划分频率区间
        bin_edges = torch.linspace(0, freq_dist.max(), self.freq_bins + 1).to(x.device)
        energy_bins = []
        for i in range(self.freq_bins):
            mask = (freq_dist >= bin_edges[i]) & (freq_dist < bin_edges[i + 1])
            energy = (magnitude * mask[None, None]).sum(dim=(-1, -2))  # [B, C]
            energy_bins.append(energy.mean(dim=1))  # [B]

        energy_profile = torch.stack(energy_bins, dim=1)  # [B, freq_bins]
        return energy_profile

    def _get_freq_grid(self, H, W):
        y = torch.arange(-H // 2, H // 2)
        x = torch.arange(-W // 2, W // 2)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1)
        return grid.float()

    def load_loss(self, logits, logits_noisy, noise_std):
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std

        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)
        p_mean = p.mean(dim=0)
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2
        return loss_load

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)
        imp_mean = importance.mean()
        imp_std = importance.std()
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2
        return loss_imp

