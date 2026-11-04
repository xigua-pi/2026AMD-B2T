"""面向 FPGA 的 H144-v2 严格递归定点 QAT 模型。

数值约束：
- 权重：逐输出通道 INT8、2 的幂 scale；
- 神经激活：逐层 INT8、固定 2 的幂 scale；
- GRU hidden：逐层 INT16、固定 2 的幂 scale，并在每个时间步回写；
- 矩阵累加器：INT32（训练时以 FP32 精确表示量化乘加，并检查无溢出）；
- reset/update：UQ0.16；candidate：Q1.15；门前激活：Q4.11；
- 输出 logits：统一 Q7.8，确保不同类别可直接比较。

所有 scale 在一次推理过程中保持不变。训练允许更新逐层 log2(scale)，但 forward
始终使用取整后的 2 的幂 scale，因此冻结后只需移位、舍入和饱和。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from compact_bci_model import CompactBrainToTextGRU


def round_ste(values: torch.Tensor) -> torch.Tensor:
    """前向使用四舍五入，反向使用恒等近似。"""

    return values + (torch.round(values) - values).detach()


def symmetric_limits(bits: int) -> Tuple[int, int]:
    maximum = (1 << (int(bits) - 1)) - 1
    return -maximum, maximum


class PowerOfTwoQuantizer(nn.Module):
    """具有固定逐层 scale 的对称定点量化器。"""

    def __init__(
        self,
        bits: int,
        initial_exponent: float,
        learnable: bool,
        signed: bool = True,
        observer_percentile: float = 0.999,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.signed = bool(signed)
        self.learnable_scale = bool(learnable)
        self.observer_percentile = float(observer_percentile)
        self.log2_scale = nn.Parameter(
            torch.tensor(float(initial_exponent), dtype=torch.float32),
            requires_grad=learnable,
        )
        self.register_buffer("observed_abs", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("observer_updates", torch.zeros(1, dtype=torch.long))
        self.observer_enabled = False
        self.bypass = False

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) + 1 if self.signed else 0

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - (1 if self.signed else 0))) - 1

    def rounded_exponent(self) -> torch.Tensor:
        if self.log2_scale.requires_grad and self.training:
            return round_ste(self.log2_scale)
        return torch.round(self.log2_scale)

    def scale(self) -> torch.Tensor:
        return torch.pow(self.log2_scale.new_tensor(2.0), self.rounded_exponent())

    @torch.no_grad()
    def _observe(self, values: torch.Tensor) -> None:
        flattened = values.detach().abs().float().reshape(-1)
        if flattened.numel() == 0:
            return
        # R1 先使用保守的 sample-max，避免长序列逐时刻执行 quantile；
        # R2 再通过可学习 clipping scale 回收动态范围。
        stride = max(1, flattened.numel() // 8192)
        sampled = flattened[::stride]
        observed = sampled.amax().clamp_min(1.0e-12)
        # 兼容工作站较旧的 PyTorch（没有 Tensor.maximum_）。
        self.observed_abs.copy_(
            torch.maximum(self.observed_abs, observed.reshape_as(self.observed_abs))
        )
        self.observer_updates.add_(1)

    @torch.no_grad()
    def freeze_from_observer(self, margin: float = 1.05) -> int:
        if int(self.observer_updates.item()) == 0:
            return int(round(float(self.log2_scale.item())))
        required_scale = float(self.observed_abs.item()) * float(margin) / max(self.qmax, 1)
        exponent = int(math.ceil(math.log2(max(required_scale, 1.0e-12))))
        self.log2_scale.fill_(float(exponent))
        return exponent

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            self._observe(values)
        if self.bypass:
            return values
        scale = self.scale().to(device=values.device, dtype=values.dtype)
        scaled = values / scale
        clipped = torch.clamp(scaled, self.qmin, self.qmax)
        quantized = round_ste(clipped)
        return quantized * scale

    @torch.no_grad()
    def integer(self, values: torch.Tensor) -> torch.Tensor:
        scale = self.scale().to(device=values.device, dtype=values.dtype)
        return torch.clamp(torch.round(values / scale), self.qmin, self.qmax).to(torch.int64)

    def specification(self) -> Dict[str, object]:
        exponent = int(round(float(self.log2_scale.detach().cpu().item())))
        return {
            "bits": self.bits,
            "signed": self.signed,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "scale_exponent": exponent,
            "scale": float(2.0**exponent),
            "clip_min": float(self.qmin * (2.0**exponent)),
            "clip_max": float(self.qmax * (2.0**exponent)),
            "learnable_during_qat": self.learnable_scale,
        }


def fake_quantize_weight_pot(
    values: torch.Tensor,
    enabled: bool,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """逐输出通道进行 INT8 + power-of-two scale 量化。"""

    reduce_dims = tuple(range(1, values.ndim))
    qmin, qmax = symmetric_limits(bits)
    maximum = values.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1.0e-12)
    exponent = torch.ceil(torch.log2(maximum / float(qmax)))
    scale = torch.pow(values.new_tensor(2.0), exponent)
    if not enabled:
        return values, scale
    integer = torch.clamp(torch.round(values / scale), qmin, qmax)
    dequantized = integer * scale
    return values + (dequantized - values).detach(), scale


def fake_quantize_batched_weight_pot(
    values: torch.Tensor,
    enabled: bool,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 ``[batch, output, input]`` 权重逐 batch、逐输出通道量化。"""

    if values.ndim != 3:
        raise ValueError(f"Expected batched 3-D weights, got {tuple(values.shape)}")
    qmin, qmax = symmetric_limits(bits)
    maximum = values.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
    exponent = torch.ceil(torch.log2(maximum / float(qmax)))
    scale = torch.pow(values.new_tensor(2.0), exponent)
    if not enabled:
        return values, scale
    integer = torch.clamp(torch.round(values / scale), qmin, qmax)
    dequantized = integer * scale
    return values + (dequantized - values).detach(), scale


def fake_quantize_bias(
    bias: torch.Tensor,
    product_scale: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """把 bias 映射到对应矩阵乘加累加器的 INT32 scale。"""

    if not enabled:
        return bias
    scale = product_scale.reshape(-1).to(device=bias.device, dtype=bias.dtype)
    qmin, qmax = -(1 << 31), (1 << 31) - 1
    integer = torch.clamp(torch.round(bias / scale), qmin, qmax)
    dequantized = integer * scale
    return bias + (dequantized - bias).detach()


class RecurrentFixedPointH144(CompactBrainToTextGRU):
    """保持 B2 拓扑不变、但 forward 严格回写 INT16 hidden 的模型。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 初始 exponent 仅用于校准前的安全启动；R1 会由 403 条 fit 数据冻结。
        self.input_q = PowerOfTwoQuantizer(8, -4, True)
        self.adapter_mid_q = PowerOfTwoQuantizer(8, -6, True)
        self.adapter_out_q = PowerOfTwoQuantizer(8, -4, True)
        self.depthwise_q = PowerOfTwoQuantizer(8, -5, True)
        self.bottleneck_q = PowerOfTwoQuantizer(8, -5, True)
        self.gru2_input_q = PowerOfTwoQuantizer(8, -7, True)
        self.hidden1_q = PowerOfTwoQuantizer(16, -15, True)
        self.hidden2_q = PowerOfTwoQuantizer(16, -15, True)

        # FPGA 固定格式，不参与 scale 学习。
        self.preact1_q = PowerOfTwoQuantizer(16, -11, False)
        self.preact2_q = PowerOfTwoQuantizer(16, -11, False)
        self.reset_q = PowerOfTwoQuantizer(16, -16, False, signed=False)
        self.update_q = PowerOfTwoQuantizer(16, -16, False, signed=False)
        self.candidate_q = PowerOfTwoQuantizer(16, -15, False)
        self.logit_q = PowerOfTwoQuantizer(16, -8, False)

        self.quantization_enabled = True
        self.activation_mode = "exact"
        self.lut_entries = 256
        self.lut_range = 8.0
        self.pwl_segments = 8

    def quantizers(self) -> Dict[str, PowerOfTwoQuantizer]:
        return {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, PowerOfTwoQuantizer)
        }

    def set_observer_mode(self, enabled: bool, bypass: bool) -> None:
        for quantizer in self.quantizers().values():
            quantizer.observer_enabled = bool(enabled)
            quantizer.bypass = bool(bypass)

    def set_scale_trainable(self, enabled: bool) -> None:
        for name, quantizer in self.quantizers().items():
            trainable = bool(enabled) and quantizer.learnable_scale
            quantizer.log2_scale.requires_grad_(trainable)

    @torch.no_grad()
    def freeze_observed_scales(self, margin: float = 1.05) -> Dict[str, int]:
        return {
            name: quantizer.freeze_from_observer(margin)
            for name, quantizer in self.quantizers().items()
            if quantizer.learnable_scale
        }

    def _activation(self, values: torch.Tensor, kind: str) -> torch.Tensor:
        if self.activation_mode == "exact":
            return torch.sigmoid(values) if kind == "sigmoid" else torch.tanh(values)
        clipped = values.clamp(-self.lut_range, self.lut_range)
        if self.activation_mode == "lut":
            points = int(self.lut_entries)
            grid = torch.linspace(
                -self.lut_range,
                self.lut_range,
                points,
                device=values.device,
                dtype=values.dtype,
            )
            table = torch.sigmoid(grid) if kind == "sigmoid" else torch.tanh(grid)
            index = torch.round(
                (clipped + self.lut_range)
                * ((points - 1) / (2.0 * self.lut_range))
            ).long().clamp(0, points - 1)
            return table[index]
        if self.activation_mode == "pwl":
            segments = int(self.pwl_segments)
            grid = torch.linspace(
                -self.lut_range,
                self.lut_range,
                segments + 1,
                device=values.device,
                dtype=values.dtype,
            )
            knots = torch.sigmoid(grid) if kind == "sigmoid" else torch.tanh(grid)
            position = (clipped + self.lut_range) * (
                segments / (2.0 * self.lut_range)
            )
            left = torch.floor(position).long().clamp(0, segments - 1)
            fraction = (position - left.to(position.dtype)).clamp(0.0, 1.0)
            return knots[left] + fraction * (knots[left + 1] - knots[left])
        raise ValueError(f"Unsupported activation mode: {self.activation_mode}")

    def _adapter_forward(self, inputs: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
        x = self.input_q(inputs)
        u = self.session_adapter.u[session_ids]
        v = self.session_adapter.v[session_ids]
        v_q, _ = fake_quantize_batched_weight_pot(v, self.quantization_enabled)
        u_q, _ = fake_quantize_batched_weight_pot(u, self.quantization_enabled)
        middle = self.adapter_mid_q(torch.bmm(x, v_q.transpose(1, 2)))
        low_rank = torch.bmm(middle, u_q.transpose(1, 2))
        # 三个分支先独立 requantize 到 adapter_out 的共同 scale，再相加。
        # 这与 FPGA 的“移位 -> 饱和 -> 加法”数据通路一致。
        residual = self.adapter_out_q(x)
        low_rank = self.adapter_out_q(low_rank)
        bias = self.adapter_out_q(self.session_adapter.bias[session_ids]).unsqueeze(1)
        output = residual + low_rank + bias
        return self.adapter_out_q(output)

    def _temporal_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal_bottleneck
        values = inputs.transpose(1, 2)
        if temporal.causal:
            values = F.pad(values, (temporal.kernel_size - 1, 0))
        else:
            radius = temporal.kernel_size // 2
            values = F.pad(values, (radius, radius))
        depthwise_weight, _ = fake_quantize_weight_pot(
            temporal.depthwise.weight, self.quantization_enabled
        )
        values = F.conv1d(
            values,
            depthwise_weight,
            bias=None,
            groups=temporal.input_size,
        )
        values = self.depthwise_q(values.transpose(1, 2)).transpose(1, 2)
        pointwise_weight, pointwise_scale = fake_quantize_weight_pot(
            temporal.pointwise.weight, self.quantization_enabled
        )
        input_scale = self.depthwise_q.scale()
        pointwise_bias = fake_quantize_bias(
            temporal.pointwise.bias,
            input_scale * pointwise_scale,
            self.quantization_enabled,
        )
        values = F.conv1d(values, pointwise_weight, bias=pointwise_bias)
        return self.bottleneck_q(values.transpose(1, 2))

    def _gru_sequence(
        self,
        gru: nn.GRU,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        start_state: torch.Tensor,
        input_quantizer: PowerOfTwoQuantizer,
        hidden_quantizer: PowerOfTwoQuantizer,
        preact_quantizer: PowerOfTwoQuantizer,
        return_trace: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        weight_ih, weight_ih_scale = fake_quantize_weight_pot(
            gru.weight_ih_l0, self.quantization_enabled
        )
        weight_hh, weight_hh_scale = fake_quantize_weight_pot(
            gru.weight_hh_l0, self.quantization_enabled
        )
        bias_ih = fake_quantize_bias(
            gru.bias_ih_l0,
            input_quantizer.scale() * weight_ih_scale,
            self.quantization_enabled,
        )
        bias_hh = fake_quantize_bias(
            gru.bias_hh_l0,
            hidden_quantizer.scale() * weight_hh_scale,
            self.quantization_enabled,
        )
        input_gates = preact_quantizer(F.linear(inputs, weight_ih, bias_ih))
        hidden = hidden_quantizer(
            start_state.repeat(1, inputs.shape[0], 1)[0]
        )
        outputs: List[torch.Tensor] = []
        trace_reset: List[torch.Tensor] = []
        trace_update: List[torch.Tensor] = []
        trace_candidate: List[torch.Tensor] = []
        trace_reset_unquantized: List[torch.Tensor] = []
        trace_update_unquantized: List[torch.Tensor] = []
        trace_candidate_unquantized: List[torch.Tensor] = []
        trace_pre_hidden: List[torch.Tensor] = []
        for time_index in range(inputs.shape[1]):
            hidden_gates = preact_quantizer(F.linear(hidden, weight_hh, bias_hh))
            input_reset, input_update, input_new = input_gates[:, time_index].chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
            reset_unquantized = self._activation(
                preact_quantizer(input_reset + hidden_reset), "sigmoid"
            )
            update_unquantized = self._activation(
                preact_quantizer(input_update + hidden_update), "sigmoid"
            )
            reset = self.reset_q(reset_unquantized)
            update = self.update_q(update_unquantized)
            reset_hidden_new = preact_quantizer(reset * hidden_new)
            candidate_unquantized = self._activation(
                preact_quantizer(input_new + reset_hidden_new), "tanh"
            )
            candidate = self.candidate_q(candidate_unquantized)
            pre_hidden = hidden_quantizer((1.0 - update) * candidate) + hidden_quantizer(
                update * hidden
            )
            quantized_hidden = hidden_quantizer(pre_hidden)
            active = (time_index < lengths).to(quantized_hidden.dtype).unsqueeze(-1)
            hidden = active * quantized_hidden + (1.0 - active) * hidden
            outputs.append(hidden)
            if return_trace:
                trace_reset.append(reset)
                trace_update.append(update)
                trace_candidate.append(candidate)
                trace_reset_unquantized.append(reset_unquantized)
                trace_update_unquantized.append(update_unquantized)
                trace_candidate_unquantized.append(candidate_unquantized)
                trace_pre_hidden.append(pre_hidden)
        trace: Dict[str, torch.Tensor] = {}
        if return_trace:
            trace = {
                "reset": torch.stack(trace_reset, dim=1),
                "update": torch.stack(trace_update, dim=1),
                "candidate": torch.stack(trace_candidate, dim=1),
                "reset_unquantized": torch.stack(trace_reset_unquantized, dim=1),
                "update_unquantized": torch.stack(trace_update_unquantized, dim=1),
                "candidate_unquantized": torch.stack(trace_candidate_unquantized, dim=1),
                "pre_hidden": torch.stack(trace_pre_hidden, dim=1),
            }
        return torch.stack(outputs, dim=1), trace

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        session_ids: torch.Tensor,
        return_trace: bool = False,
    ):
        values = self._adapter_forward(inputs, session_ids)
        values = self._temporal_forward(values)
        values = self.dropout(values)
        output1, trace1 = self._gru_sequence(
            self.gru1,
            values,
            input_lengths,
            self.start_state_1,
            self.bottleneck_q,
            self.hidden1_q,
            self.preact1_q,
            return_trace,
        )
        output1 = self.dropout(output1)
        downsampled = self.gru2_input_q(output1[:, :: self.skip_len, :])
        down_lengths = torch.div(
            input_lengths + self.skip_len - 1,
            self.skip_len,
            rounding_mode="floor",
        ).clamp_min(1)
        output2, trace2 = self._gru_sequence(
            self.gru2,
            downsampled,
            down_lengths,
            self.start_state_2,
            self.gru2_input_q,
            self.hidden2_q,
            self.preact2_q,
            return_trace,
        )
        output2 = self.dropout(output2)
        readout_weight, readout_scale = fake_quantize_weight_pot(
            self.readout.weight, self.quantization_enabled
        )
        readout_bias = fake_quantize_bias(
            self.readout.bias,
            self.hidden2_q.scale() * readout_scale,
            self.quantization_enabled,
        )
        logits_pre_quant = F.linear(output2, readout_weight, readout_bias)
        logits_down = self.logit_q(logits_pre_quant)
        logits = torch.repeat_interleave(logits_down, repeats=self.skip_len, dim=1)
        logits = logits[:, : inputs.shape[1]]
        if not return_trace:
            return logits, input_lengths
        return logits, input_lengths, {
            "gru1": {**trace1, "hidden": output1},
            "gru2": {**trace2, "hidden": output2},
            "logits_pre_quant": logits_pre_quant,
            "logits_down": logits_down,
            "logits": logits,
        }

    def fixed_point_specification(self) -> Dict[str, object]:
        return {
            "rounding": "round_to_nearest_ties_to_even",
            "saturation": "symmetric for signed tensors",
            "accumulator_bits": 32,
            "weight_bits": 8,
            "weight_scale": "per-output-channel power-of-two",
            "activation_scale": "per-layer power-of-two, invariant over time",
            "hidden_feedback": "quantized INT16 h_t is the only state used at t+1",
            "gate_order": ["reset", "update", "new"],
            "activation_mode": self.activation_mode,
            "lut_entries": self.lut_entries if self.activation_mode == "lut" else None,
            "lut_range": self.lut_range,
            "pwl_segments": self.pwl_segments if self.activation_mode == "pwl" else None,
            "quantizers": {
                name: quantizer.specification()
                for name, quantizer in self.quantizers().items()
            },
        }

    def hardware_profile(self) -> Dict[str, float]:
        """量化 scale 是配置常数，不计入神经网络参数量。"""

        profile = super().hardware_profile()
        scale_parameter_count = sum(
            quantizer.log2_scale.numel() for quantizer in self.quantizers().values()
        )
        profile["parameters"] -= float(scale_parameter_count)
        profile["fixed_scale_constants"] = float(scale_parameter_count)
        return profile
