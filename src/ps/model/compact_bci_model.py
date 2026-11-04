from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from paper_v5_pipeline import N_CHAR_CLASSES


class LowRankSessionAdapter(nn.Module):
    """以 I + U_s V_s 表示每个会话的 192x192 仿射校准矩阵。"""

    def __init__(self, input_size: int, num_sessions: int, rank: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.num_sessions = int(num_sessions)
        self.rank = int(rank)

        # U 随机、V 置零：初始映射严格为恒等，同时第一步仍可向 V 传播梯度。
        self.u = nn.Parameter(torch.empty(num_sessions, input_size, rank))
        self.v = nn.Parameter(torch.zeros(num_sessions, rank, input_size))
        self.bias = nn.Parameter(torch.zeros(num_sessions, input_size))
        nn.init.normal_(self.u, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
        u = self.u[session_ids]
        v = self.v[session_ids]
        low_rank = torch.bmm(torch.bmm(inputs, v.transpose(1, 2)), u.transpose(1, 2))
        return inputs + low_rank + self.bias[session_ids].unsqueeze(1)

    @torch.no_grad()
    def initialize_from_dense(self, dense_weight: torch.Tensor, dense_bias: torch.Tensor) -> None:
        """用截断 SVD 拟合已有会话矩阵，保留原模型中最主要的会话差异。"""

        if tuple(dense_weight.shape) != (self.num_sessions, self.input_size, self.input_size):
            raise ValueError(f"Unexpected dense session weight shape: {tuple(dense_weight.shape)}")
        eye = torch.eye(self.input_size, dtype=dense_weight.dtype, device=dense_weight.device)
        for session_id in range(self.num_sessions):
            residual = dense_weight[session_id] - eye
            left, singular, right_h = torch.linalg.svd(residual, full_matrices=False)
            singular_root = singular[: self.rank].clamp_min(0).sqrt()
            self.u[session_id].copy_(left[:, : self.rank] * singular_root.unsqueeze(0))
            self.v[session_id].copy_(singular_root.unsqueeze(1) * right_h[: self.rank, :])
        self.bias.copy_(dense_bias)


class SeparableTemporalBottleneck(nn.Module):
    """逐通道时域滤波后，以 1x1 卷积把 192 维压缩到较小瓶颈。"""

    def __init__(self, input_size: int, bottleneck_size: int, kernel_size: int, causal: bool) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.input_size = int(input_size)
        self.bottleneck_size = int(bottleneck_size)
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        self.depthwise = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            groups=input_size,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_size, bottleneck_size, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 以 sigma=2 的高斯核起步，训练时允许每个通道形成不同的时域滤波器。
        radius = self.kernel_size // 2
        positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-(positions**2) / (2.0 * 2.0**2))
        kernel = kernel / kernel.sum()
        with torch.no_grad():
            self.depthwise.weight.copy_(kernel.view(1, 1, -1).repeat(self.input_size, 1, 1))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=5**0.5)
        if self.pointwise.bias is not None:
            nn.init.zeros_(self.pointwise.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        if self.causal:
            x = torch.nn.functional.pad(x, (self.kernel_size - 1, 0))
        else:
            radius = self.kernel_size // 2
            x = torch.nn.functional.pad(x, (radius, radius))
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x.transpose(1, 2)

    @torch.no_grad()
    def initialize_projection_from_teacher_gru(self, teacher_weight_ih: torch.Tensor) -> None:
        """以教师 GRU 输入权重的主右奇异向量初始化 192->B 投影。"""

        if teacher_weight_ih.ndim != 2 or teacher_weight_ih.shape[1] != self.input_size:
            raise ValueError(f"Unexpected teacher GRU input weight shape: {tuple(teacher_weight_ih.shape)}")
        _left, _singular, right_h = torch.linalg.svd(teacher_weight_ih.float(), full_matrices=False)
        projection = right_h[: self.bottleneck_size, :]
        self.pointwise.weight.copy_(projection.to(self.pointwise.weight).unsqueeze(-1))
        if self.pointwise.bias is not None:
            self.pointwise.bias.zero_()


class CompactBrainToTextGRU(nn.Module):
    """面向 FPGA 的低秩会话适配 + 可分离卷积 + 双层小型 GRU。"""

    def __init__(
        self,
        input_size: int = 192,
        bottleneck_size: int = 64,
        hidden_size: int = 128,
        num_sessions: int = 10,
        adapter_rank: int = 16,
        temporal_kernel: int = 13,
        causal_conv: bool = False,
        skip_len: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.bottleneck_size = int(bottleneck_size)
        self.hidden_size = int(hidden_size)
        self.num_sessions = int(num_sessions)
        self.adapter_rank = int(adapter_rank)
        self.temporal_kernel = int(temporal_kernel)
        self.causal_conv = bool(causal_conv)
        self.skip_len = int(skip_len)

        self.session_adapter = LowRankSessionAdapter(input_size, num_sessions, adapter_rank)
        self.temporal_bottleneck = SeparableTemporalBottleneck(
            input_size=input_size,
            bottleneck_size=bottleneck_size,
            kernel_size=temporal_kernel,
            causal=causal_conv,
        )
        self.dropout = nn.Dropout(dropout)
        self.gru1 = nn.GRU(bottleneck_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, N_CHAR_CLASSES + 1)
        self.start_state_1 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.start_state_2 = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def _run_gru(
        self,
        gru: nn.GRU,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        start_state: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        initial = start_state.repeat(1, inputs.shape[0], 1)
        packed_output, _ = gru(packed, initial)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        session_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.session_adapter(inputs, session_ids)
        x = self.temporal_bottleneck(x)
        x = self.dropout(x)

        out1 = self._run_gru(self.gru1, x, input_lengths, self.start_state_1)
        out1 = self.dropout(out1)
        downsampled = out1[:, :: self.skip_len, :]
        down_lengths = torch.div(
            input_lengths + (self.skip_len - 1), self.skip_len, rounding_mode="floor"
        ).clamp_min(1)
        out2 = self._run_gru(self.gru2, downsampled, down_lengths, self.start_state_2)
        out2 = self.dropout(out2)

        logits_down = self.readout(out2)
        logits = torch.repeat_interleave(logits_down, repeats=self.skip_len, dim=1)
        return logits[:, : inputs.shape[1], :], input_lengths

    @torch.no_grad()
    def initialize_from_teacher(self, teacher: nn.Module) -> None:
        self.session_adapter.initialize_from_dense(teacher.session_weight, teacher.session_bias)
        self.temporal_bottleneck.initialize_projection_from_teacher_gru(teacher.gru1.weight_ih_l0)

    def hardware_profile(self) -> Dict[str, float]:
        """给出每个 20 ms 输入步的稠密 MAC 解析估计，便于和基线统一比较。"""

        i = self.input_size
        b = self.bottleneck_size
        h = self.hidden_size
        r = self.adapter_rank
        s = self.skip_len
        output_size = N_CHAR_CLASSES + 1

        adapter_mac = 2 * i * r
        temporal_mac = i * self.temporal_kernel + i * b
        gru1_mac = 3 * h * (b + h)
        gru2_mac = 3 * h * (h + h) / s
        readout_mac = h * output_size / s
        total_mac = adapter_mac + temporal_mac + gru1_mac + gru2_mac + readout_mac
        return {
            "parameters": float(sum(parameter.numel() for parameter in self.parameters())),
            "mac_per_20ms_step": float(total_mac),
            "adapter_mac": float(adapter_mac),
            "temporal_mac": float(temporal_mac),
            "gru1_mac": float(gru1_mac),
            "gru2_mac_amortized": float(gru2_mac),
            "readout_mac_amortized": float(readout_mac),
        }

    def export_config(self) -> Dict[str, object]:
        return {
            "input_size": self.input_size,
            "bottleneck_size": self.bottleneck_size,
            "hidden_size": self.hidden_size,
            "num_sessions": self.num_sessions,
            "adapter_rank": self.adapter_rank,
            "temporal_kernel": self.temporal_kernel,
            "causal_conv": self.causal_conv,
            "skip_len": self.skip_len,
            "dropout": self.dropout.p,
        }
