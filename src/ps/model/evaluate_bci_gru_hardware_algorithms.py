from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from brain_to_text_main_v5 import compute_error_rates, edit_distance, render_transcript
from paper_v5_pipeline import (
    PaperBrainToTextV5,
    PaperV5EvalDataset,
    SentenceLabelRecord,
    collate_variable_paper_batch,
    decode_paper_v5_outputs,
    prepare_paper_v5_label_cache,
)


@dataclass(frozen=True)
class FixedPointSpec:
    bits: int = 16
    frac_bits: int = 8
    enabled: bool = False

    @property
    def min_int(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def max_int(self) -> int:
        return (1 << (self.bits - 1)) - 1


@dataclass(frozen=True)
class VariantConfig:
    name: str
    fixed_point: FixedPointSpec = FixedPointSpec(enabled=False)
    input_spec: Optional[FixedPointSpec] = None
    weight_spec: Optional[FixedPointSpec] = None
    preact_spec: Optional[FixedPointSpec] = None
    r_spec: Optional[FixedPointSpec] = None
    z_spec: Optional[FixedPointSpec] = None
    n_spec: Optional[FixedPointSpec] = None
    h_spec: Optional[FixedPointSpec] = None
    readout_spec: Optional[FixedPointSpec] = None
    activation_mode: str = "exact"
    lut_size: int = 1024
    lut_range: float = 8.0
    delta_input_threshold: float = 0.0
    delta_hidden_threshold: float = 0.0
    use_delta_gru: bool = False
    blank_aware: bool = False
    blank_source: str = "baseline_start"
    blank_start_threshold: float = 0.3
    blank_context_radius: int = 15
    blank_threshold_scale: float = 2.0
    char_start_threshold_scale: float = 0.5
    causal_control_delay: int = 1
    quant_profile: str = "float"


@dataclass
class HardwareStats:
    dense_macs: int = 0
    actual_macs: int = 0
    dense_bops: int = 0
    actual_bops: int = 0
    dense_weight_reads: int = 0
    actual_weight_reads: int = 0
    dense_weight_read_bits: int = 0
    actual_weight_read_bits: int = 0
    sigmoid_calls: int = 0
    tanh_calls: int = 0
    input_active: int = 0
    input_total: int = 0
    hidden_active: int = 0
    hidden_total: int = 0
    blank_active_ratio_sum: float = 0.0
    blank_active_ratio_count: int = 0
    notes: List[str] = field(default_factory=list)

    def add_dense_matvec(self, rows: int, cols: int, activation_bits: int = 32, weight_bits: int = 32) -> None:
        macs = int(rows) * int(cols)
        self.dense_macs += macs
        self.actual_macs += macs
        self.dense_bops += macs * int(activation_bits) * int(weight_bits)
        self.actual_bops += macs * int(activation_bits) * int(weight_bits)
        self.dense_weight_reads += macs
        self.actual_weight_reads += macs
        self.dense_weight_read_bits += macs * int(weight_bits)
        self.actual_weight_read_bits += macs * int(weight_bits)

    def add_dense_matvec_batch(
        self,
        steps: int,
        rows: int,
        cols: int,
        activation_bits: int = 32,
        weight_bits: int = 32,
    ) -> None:
        for _ in range(int(steps)):
            self.add_dense_matvec(rows, cols, activation_bits=activation_bits, weight_bits=weight_bits)

    def add_delta_matvec(
        self,
        rows: int,
        cols: int,
        active_cols: int,
        input_side: bool,
        activation_bits: int = 32,
        weight_bits: int = 32,
    ) -> None:
        rows = int(rows)
        cols = int(cols)
        active_cols = int(active_cols)
        dense = rows * cols
        actual = rows * active_cols
        self.dense_macs += dense
        self.actual_macs += actual
        self.dense_bops += dense * int(activation_bits) * int(weight_bits)
        self.actual_bops += actual * int(activation_bits) * int(weight_bits)
        self.dense_weight_reads += dense
        self.actual_weight_reads += actual
        self.dense_weight_read_bits += dense * int(weight_bits)
        self.actual_weight_read_bits += actual * int(weight_bits)
        if input_side:
            self.input_active += active_cols
            self.input_total += cols
        else:
            self.hidden_active += active_cols
            self.hidden_total += cols

    def merge(self, other: "HardwareStats") -> None:
        self.dense_macs += other.dense_macs
        self.actual_macs += other.actual_macs
        self.dense_bops += other.dense_bops
        self.actual_bops += other.actual_bops
        self.dense_weight_reads += other.dense_weight_reads
        self.actual_weight_reads += other.actual_weight_reads
        self.dense_weight_read_bits += other.dense_weight_read_bits
        self.actual_weight_read_bits += other.actual_weight_read_bits
        self.sigmoid_calls += other.sigmoid_calls
        self.tanh_calls += other.tanh_calls
        self.input_active += other.input_active
        self.input_total += other.input_total
        self.hidden_active += other.hidden_active
        self.hidden_total += other.hidden_total
        self.blank_active_ratio_sum += other.blank_active_ratio_sum
        self.blank_active_ratio_count += other.blank_active_ratio_count
        self.notes.extend(other.notes)

    def add_blank_active_ratio(self, ratio: float) -> None:
        self.blank_active_ratio_sum += float(ratio)
        self.blank_active_ratio_count += 1

    def summary(self) -> Dict[str, float]:
        mac_reduction = 1.0 - (self.actual_macs / self.dense_macs) if self.dense_macs else 0.0
        bop_reduction = 1.0 - (self.actual_bops / self.dense_bops) if self.dense_bops else 0.0
        read_reduction = (
            1.0 - (self.actual_weight_reads / self.dense_weight_reads) if self.dense_weight_reads else 0.0
        )
        read_bit_reduction = (
            1.0 - (self.actual_weight_read_bits / self.dense_weight_read_bits)
            if self.dense_weight_read_bits
            else 0.0
        )
        input_active_ratio = self.input_active / self.input_total if self.input_total else 1.0
        hidden_active_ratio = self.hidden_active / self.hidden_total if self.hidden_total else 1.0
        blank_active_ratio = (
            self.blank_active_ratio_sum / self.blank_active_ratio_count if self.blank_active_ratio_count else 0.0
        )
        return {
            "dense_macs": float(self.dense_macs),
            "actual_macs": float(self.actual_macs),
            "mac_reduction": float(mac_reduction),
            "dense_bops": float(self.dense_bops),
            "actual_bops": float(self.actual_bops),
            "bop_reduction": float(bop_reduction),
            "dense_weight_reads": float(self.dense_weight_reads),
            "actual_weight_reads": float(self.actual_weight_reads),
            "weight_read_reduction": float(read_reduction),
            "dense_weight_read_bits": float(self.dense_weight_read_bits),
            "actual_weight_read_bits": float(self.actual_weight_read_bits),
            "weight_read_bit_reduction": float(read_bit_reduction),
            "sigmoid_calls": float(self.sigmoid_calls),
            "tanh_calls": float(self.tanh_calls),
            "input_active_ratio": float(input_active_ratio),
            "hidden_active_ratio": float(hidden_active_ratio),
            "blank_active_ratio": float(blank_active_ratio),
        }


class LutActivations:
    def __init__(self, size: int = 1024, value_range: float = 8.0, device: torch.device | str = "cpu") -> None:
        self.size = int(size)
        self.value_range = float(value_range)
        self.device = torch.device(device)
        xs = torch.linspace(-self.value_range, self.value_range, self.size, device=self.device)
        self.sigmoid_table = torch.sigmoid(xs)
        self.tanh_table = torch.tanh(xs)

    def _lookup(self, x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        clipped = x.clamp(-self.value_range, self.value_range)
        scaled = (clipped + self.value_range) * ((self.size - 1) / (2.0 * self.value_range))
        idx = torch.round(scaled).to(torch.long).clamp(0, self.size - 1)
        return table[idx]

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return self._lookup(x, self.sigmoid_table)

    def tanh(self, x: torch.Tensor) -> torch.Tensor:
        return self._lookup(x, self.tanh_table)


def resolve_existing_path(preferred_path: str | None, fallback_path: str | None) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path
    return preferred_path or fallback_path


def find_default_training_steps_root(repo_root: Path) -> str | None:
    candidates = [
        repo_root
        / "dataset"
        / "doi_10_5061_dryad_wh70rxwmv__v20211015"
        / "handwritingBCIData"
        / "handwritingBCIData"
        / "RNNTrainingSteps",
        repo_root / "RNNTrainingSteps",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def load_existing_label_cache(cache_path: str) -> Dict[str, object]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = []
    for item in payload["records"]:
        if isinstance(item, SentenceLabelRecord):
            records.append(item)
        else:
            records.append(SentenceLabelRecord.from_serializable(item))
    payload["records"] = records
    return payload


def fixed_quantize(x: torch.Tensor, spec: FixedPointSpec) -> torch.Tensor:
    if not spec.enabled:
        return x
    scale = float(1 << spec.frac_bits)
    q = torch.round(x * scale).clamp(spec.min_int, spec.max_int)
    return q / scale


def qspec(variant: VariantConfig, name: str) -> FixedPointSpec:
    override = getattr(variant, f"{name}_spec")
    return override if override is not None else variant.fixed_point


def spec_label(spec: FixedPointSpec) -> str:
    if not spec.enabled:
        return "float"
    return f"Q{spec.bits}.{spec.frac_bits}"


def spec_bits(spec: FixedPointSpec) -> int:
    return int(spec.bits) if spec.enabled else 32


def quant_profile_summary(variant: VariantConfig) -> str:
    fields = ["input", "weight", "preact", "r", "z", "n", "h", "readout"]
    parts = [f"base={spec_label(variant.fixed_point)}"]
    for field_name in fields:
        parts.append(f"{field_name}={spec_label(qspec(variant, field_name))}")
    return ",".join(parts)


def apply_gate_activation(
    x: torch.Tensor,
    kind: str,
    gate: str,
    variant: VariantConfig,
    lut: Optional[LutActivations],
    stats: HardwareStats,
) -> torch.Tensor:
    if kind == "sigmoid":
        stats.sigmoid_calls += int(x.numel())
        out = lut.sigmoid(x) if variant.activation_mode == "lut" and lut is not None else torch.sigmoid(x)
    elif kind == "tanh":
        stats.tanh_calls += int(x.numel())
        out = lut.tanh(x) if variant.activation_mode == "lut" and lut is not None else torch.tanh(x)
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
    return fixed_quantize(out, qspec(variant, gate))


def threshold_at(base_threshold: float, schedule: Optional[torch.Tensor], t: int) -> float:
    if schedule is None:
        return float(base_threshold)
    if schedule.numel() == 0:
        return float(base_threshold)
    return float(schedule[min(int(t), int(schedule.numel()) - 1)].item())


def build_activity_mask(
    length: int,
    variant: VariantConfig,
    device: torch.device,
    baseline_logits: Optional[torch.Tensor] = None,
    char_start_target: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not variant.blank_aware:
        return None
    if variant.blank_source == "target" and char_start_target is not None:
        start_score = char_start_target[:length].to(device=device, dtype=torch.float32)
    elif baseline_logits is not None:
        start_score = torch.sigmoid(baseline_logits[:length, -1].to(device=device, dtype=torch.float32))
    elif char_start_target is not None:
        start_score = char_start_target[:length].to(device=device, dtype=torch.float32)
    else:
        return None

    active = start_score > float(variant.blank_start_threshold)
    radius = max(0, int(variant.blank_context_radius))
    if radius > 0 and active.numel() > 0:
        pooled = F.max_pool1d(
            active.to(torch.float32).view(1, 1, -1),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )
        active = pooled.view(-1) > 0
    return active[:length]


def make_threshold_schedules(
    length: int,
    variant: VariantConfig,
    device: torch.device,
    baseline_logits: Optional[torch.Tensor] = None,
    char_start_target: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
    active_mask = build_activity_mask(
        length=length,
        variant=variant,
        device=device,
        baseline_logits=baseline_logits,
        char_start_target=char_start_target,
    )
    if active_mask is None:
        return None, None, 0.0

    blank_scale = float(variant.blank_threshold_scale)
    start_scale = float(variant.char_start_threshold_scale)
    input_base = float(variant.delta_input_threshold)
    hidden_base = float(variant.delta_hidden_threshold)
    input_schedule = torch.where(
        active_mask,
        torch.full((length,), input_base * start_scale, device=device),
        torch.full((length,), input_base * blank_scale, device=device),
    )
    hidden_schedule = torch.where(
        active_mask,
        torch.full((length,), hidden_base * start_scale, device=device),
        torch.full((length,), hidden_base * blank_scale, device=device),
    )
    return input_schedule, hidden_schedule, float(active_mask.to(torch.float32).mean().item())


def downsample_threshold_schedule(
    schedule: Optional[torch.Tensor],
    skip_len: int,
    down_length: int,
) -> Optional[torch.Tensor]:
    if schedule is None:
        return None
    values: List[torch.Tensor] = []
    for idx in range(int(down_length)):
        start = int(idx) * int(skip_len)
        end = min(int(schedule.numel()), start + int(skip_len))
        if start >= int(schedule.numel()):
            values.append(schedule[-1])
        else:
            values.append(torch.min(schedule[start:end]))
    return torch.stack(values, dim=0) if values else schedule[:0]


def dense_gru_sequence(
    x_seq: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    h0: torch.Tensor,
    variant: VariantConfig,
    lut: Optional[LutActivations],
    stats: HardwareStats,
) -> torch.Tensor:
    time_steps = x_seq.shape[0]
    hidden_size = h0.numel()
    h_prev = fixed_quantize(h0, qspec(variant, "h"))
    outputs: List[torch.Tensor] = []
    weight_ih = fixed_quantize(weight_ih, qspec(variant, "weight"))
    weight_hh = fixed_quantize(weight_hh, qspec(variant, "weight"))
    bias_ih = fixed_quantize(bias_ih, qspec(variant, "preact"))
    bias_hh = fixed_quantize(bias_hh, qspec(variant, "preact"))
    x_seq_q = fixed_quantize(x_seq, qspec(variant, "input"))
    if time_steps > 0:
        stats.add_dense_matvec_batch(
            time_steps,
            weight_ih.shape[0],
            weight_ih.shape[1],
            activation_bits=spec_bits(qspec(variant, "input")),
            weight_bits=spec_bits(qspec(variant, "weight")),
        )
    gi_all = fixed_quantize(F.linear(x_seq_q, weight_ih, bias_ih), qspec(variant, "preact"))

    for t in range(time_steps):
        stats.add_dense_matvec(
            weight_hh.shape[0],
            weight_hh.shape[1],
            activation_bits=spec_bits(qspec(variant, "h")),
            weight_bits=spec_bits(qspec(variant, "weight")),
        )
        gi = gi_all[t]
        gh = F.linear(h_prev, weight_hh, bias_hh)
        gh = fixed_quantize(gh, qspec(variant, "preact"))

        i_r, i_z, i_n = gi.chunk(3, dim=0)
        h_r, h_z, h_n = gh.chunk(3, dim=0)
        r = apply_gate_activation(
            fixed_quantize(i_r + h_r, qspec(variant, "preact")),
            "sigmoid",
            "r",
            variant,
            lut,
            stats,
        )
        z = apply_gate_activation(
            fixed_quantize(i_z + h_z, qspec(variant, "preact")),
            "sigmoid",
            "z",
            variant,
            lut,
            stats,
        )
        n_pre = fixed_quantize(i_n + fixed_quantize(r * h_n, qspec(variant, "preact")), qspec(variant, "preact"))
        n = apply_gate_activation(n_pre, "tanh", "n", variant, lut, stats)
        h = fixed_quantize(
            fixed_quantize((1.0 - z) * n, qspec(variant, "h"))
            + fixed_quantize(z * h_prev, qspec(variant, "h")),
            qspec(variant, "h"),
        )
        outputs.append(h)
        h_prev = h

    return torch.stack(outputs, dim=0)


def delta_gru_sequence(
    x_seq: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    h0: torch.Tensor,
    variant: VariantConfig,
    lut: Optional[LutActivations],
    stats: HardwareStats,
    input_threshold_schedule: Optional[torch.Tensor] = None,
    hidden_threshold_schedule: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    time_steps = x_seq.shape[0]
    h_prev = fixed_quantize(h0, qspec(variant, "h"))
    outputs: List[torch.Tensor] = []
    weight_ih = fixed_quantize(weight_ih, qspec(variant, "weight"))
    weight_hh = fixed_quantize(weight_hh, qspec(variant, "weight"))
    bias_ih = fixed_quantize(bias_ih, qspec(variant, "preact"))
    bias_hh = fixed_quantize(bias_hh, qspec(variant, "preact"))

    x_ref = torch.zeros_like(x_seq[0])
    h_ref = torch.zeros_like(h_prev)
    gi = bias_ih.clone()
    gh = bias_hh.clone()

    for t in range(time_steps):
        input_threshold = threshold_at(variant.delta_input_threshold, input_threshold_schedule, t)
        hidden_threshold = threshold_at(variant.delta_hidden_threshold, hidden_threshold_schedule, t)
        x_t = fixed_quantize(x_seq[t], qspec(variant, "input"))
        delta_x = x_t - x_ref
        active_x = torch.abs(delta_x) > input_threshold
        if t == 0:
            active_x = torch.ones_like(active_x, dtype=torch.bool)
        active_x_count = int(active_x.sum().item())
        stats.add_delta_matvec(
            weight_ih.shape[0],
            weight_ih.shape[1],
            active_x_count,
            input_side=True,
            activation_bits=spec_bits(qspec(variant, "input")),
            weight_bits=spec_bits(qspec(variant, "weight")),
        )
        if active_x_count > 0:
            gi = gi + torch.mv(weight_ih[:, active_x], delta_x[active_x])
            x_ref = x_ref.clone()
            x_ref[active_x] = x_t[active_x]
        gi = fixed_quantize(gi, qspec(variant, "preact"))

        delta_h = h_prev - h_ref
        active_h = torch.abs(delta_h) > hidden_threshold
        if t == 0:
            active_h = torch.ones_like(active_h, dtype=torch.bool)
        active_h_count = int(active_h.sum().item())
        stats.add_delta_matvec(
            weight_hh.shape[0],
            weight_hh.shape[1],
            active_h_count,
            input_side=False,
            activation_bits=spec_bits(qspec(variant, "h")),
            weight_bits=spec_bits(qspec(variant, "weight")),
        )
        if active_h_count > 0:
            gh = gh + torch.mv(weight_hh[:, active_h], delta_h[active_h])
            h_ref = h_ref.clone()
            h_ref[active_h] = h_prev[active_h]
        gh = fixed_quantize(gh, qspec(variant, "preact"))

        i_r, i_z, i_n = gi.chunk(3, dim=0)
        h_r, h_z, h_n = gh.chunk(3, dim=0)
        r = apply_gate_activation(
            fixed_quantize(i_r + h_r, qspec(variant, "preact")),
            "sigmoid",
            "r",
            variant,
            lut,
            stats,
        )
        z = apply_gate_activation(
            fixed_quantize(i_z + h_z, qspec(variant, "preact")),
            "sigmoid",
            "z",
            variant,
            lut,
            stats,
        )
        n_pre = fixed_quantize(i_n + fixed_quantize(r * h_n, qspec(variant, "preact")), qspec(variant, "preact"))
        n = apply_gate_activation(n_pre, "tanh", "n", variant, lut, stats)
        h = fixed_quantize(
            fixed_quantize((1.0 - z) * n, qspec(variant, "h"))
            + fixed_quantize(z * h_prev, qspec(variant, "h")),
            qspec(variant, "h"),
        )
        outputs.append(h)
        h_prev = h

    return torch.stack(outputs, dim=0)


class GruStepRuntime:
    def __init__(
        self,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
        h0: torch.Tensor,
        variant: VariantConfig,
        lut: Optional[LutActivations],
        stats: HardwareStats,
    ) -> None:
        self.weight_ih = fixed_quantize(weight_ih, qspec(variant, "weight"))
        self.weight_hh = fixed_quantize(weight_hh, qspec(variant, "weight"))
        self.bias_ih = fixed_quantize(bias_ih, qspec(variant, "preact"))
        self.bias_hh = fixed_quantize(bias_hh, qspec(variant, "preact"))
        self.h_prev = fixed_quantize(h0, qspec(variant, "h"))
        self.variant = variant
        self.lut = lut
        self.stats = stats
        self.x_ref: Optional[torch.Tensor] = None
        self.h_ref = torch.zeros_like(self.h_prev)
        self.gi = self.bias_ih.clone()
        self.gh = self.bias_hh.clone()
        self.step_index = 0

    def _apply_gru_update(self, gi: torch.Tensor, gh: torch.Tensor) -> torch.Tensor:
        i_r, i_z, i_n = gi.chunk(3, dim=0)
        h_r, h_z, h_n = gh.chunk(3, dim=0)
        r = apply_gate_activation(
            fixed_quantize(i_r + h_r, qspec(self.variant, "preact")),
            "sigmoid",
            "r",
            self.variant,
            self.lut,
            self.stats,
        )
        z = apply_gate_activation(
            fixed_quantize(i_z + h_z, qspec(self.variant, "preact")),
            "sigmoid",
            "z",
            self.variant,
            self.lut,
            self.stats,
        )
        n_pre = fixed_quantize(
            i_n + fixed_quantize(r * h_n, qspec(self.variant, "preact")),
            qspec(self.variant, "preact"),
        )
        n = apply_gate_activation(n_pre, "tanh", "n", self.variant, self.lut, self.stats)
        h = fixed_quantize(
            fixed_quantize((1.0 - z) * n, qspec(self.variant, "h"))
            + fixed_quantize(z * self.h_prev, qspec(self.variant, "h")),
            qspec(self.variant, "h"),
        )
        self.h_prev = h
        return h

    def step_dense(self, x_t: torch.Tensor) -> torch.Tensor:
        x_t = fixed_quantize(x_t, qspec(self.variant, "input"))
        self.stats.add_dense_matvec(
            self.weight_ih.shape[0],
            self.weight_ih.shape[1],
            activation_bits=spec_bits(qspec(self.variant, "input")),
            weight_bits=spec_bits(qspec(self.variant, "weight")),
        )
        self.stats.add_dense_matvec(
            self.weight_hh.shape[0],
            self.weight_hh.shape[1],
            activation_bits=spec_bits(qspec(self.variant, "h")),
            weight_bits=spec_bits(qspec(self.variant, "weight")),
        )
        gi = fixed_quantize(F.linear(x_t, self.weight_ih, self.bias_ih), qspec(self.variant, "preact"))
        gh = fixed_quantize(F.linear(self.h_prev, self.weight_hh, self.bias_hh), qspec(self.variant, "preact"))
        self.step_index += 1
        return self._apply_gru_update(gi, gh)

    def step_delta(self, x_t: torch.Tensor, input_threshold: float, hidden_threshold: float) -> torch.Tensor:
        x_t = fixed_quantize(x_t, qspec(self.variant, "input"))
        if self.x_ref is None:
            self.x_ref = torch.zeros_like(x_t)

        delta_x = x_t - self.x_ref
        active_x = torch.abs(delta_x) > float(input_threshold)
        if self.step_index == 0:
            active_x = torch.ones_like(active_x, dtype=torch.bool)
        active_x_count = int(active_x.sum().item())
        self.stats.add_delta_matvec(
            self.weight_ih.shape[0],
            self.weight_ih.shape[1],
            active_x_count,
            input_side=True,
            activation_bits=spec_bits(qspec(self.variant, "input")),
            weight_bits=spec_bits(qspec(self.variant, "weight")),
        )
        if active_x_count > 0:
            self.gi = self.gi + torch.mv(self.weight_ih[:, active_x], delta_x[active_x])
            self.x_ref = self.x_ref.clone()
            self.x_ref[active_x] = x_t[active_x]
        self.gi = fixed_quantize(self.gi, qspec(self.variant, "preact"))

        delta_h = self.h_prev - self.h_ref
        active_h = torch.abs(delta_h) > float(hidden_threshold)
        if self.step_index == 0:
            active_h = torch.ones_like(active_h, dtype=torch.bool)
        active_h_count = int(active_h.sum().item())
        self.stats.add_delta_matvec(
            self.weight_hh.shape[0],
            self.weight_hh.shape[1],
            active_h_count,
            input_side=False,
            activation_bits=spec_bits(qspec(self.variant, "h")),
            weight_bits=spec_bits(qspec(self.variant, "weight")),
        )
        if active_h_count > 0:
            self.gh = self.gh + torch.mv(self.weight_hh[:, active_h], delta_h[active_h])
            self.h_ref = self.h_ref.clone()
            self.h_ref[active_h] = self.h_prev[active_h]
        self.gh = fixed_quantize(self.gh, qspec(self.variant, "preact"))
        self.step_index += 1
        return self._apply_gru_update(self.gi, self.gh)

    def step(self, x_t: torch.Tensor, input_threshold: float, hidden_threshold: float) -> torch.Tensor:
        if self.variant.use_delta_gru:
            return self.step_delta(x_t, input_threshold, hidden_threshold)
        return self.step_dense(x_t)


def causal_blank_gru_forward(
    model: PaperBrainToTextV5,
    x: torch.Tensor,
    length: int,
    variant: VariantConfig,
    lut: Optional[LutActivations],
    stats: HardwareStats,
    device: torch.device,
) -> torch.Tensor:
    h0_1 = model.start_state_1[0, 0].to(device)
    h0_2 = model.start_state_2[0, 0].to(device)
    runner1 = GruStepRuntime(
        model.gru1.weight_ih_l0.to(device),
        model.gru1.weight_hh_l0.to(device),
        model.gru1.bias_ih_l0.to(device),
        model.gru1.bias_hh_l0.to(device),
        h0_1,
        variant,
        lut,
        stats,
    )
    runner2 = GruStepRuntime(
        model.gru2.weight_ih_l0.to(device),
        model.gru2.weight_hh_l0.to(device),
        model.gru2.bias_ih_l0.to(device),
        model.gru2.bias_hh_l0.to(device),
        h0_2,
        variant,
        lut,
        stats,
    )
    readout_weight = fixed_quantize(model.readout.weight.to(device), qspec(variant, "readout"))
    readout_bias = fixed_quantize(model.readout.bias.to(device), qspec(variant, "readout"))

    active_count = max(0, int(variant.blank_context_radius))
    pending_events: List[int] = []
    active_steps = 0
    logits_down: List[torch.Tensor] = []
    delay = max(0, int(variant.causal_control_delay))

    for t in range(int(length)):
        due_now = [event for event in pending_events if event <= t]
        if due_now:
            active_count = max(active_count, int(variant.blank_context_radius))
        pending_events = [event for event in pending_events if event > t]

        is_active = active_count > 0
        if is_active:
            active_steps += 1
        scale = float(variant.char_start_threshold_scale if is_active else variant.blank_threshold_scale)
        input_threshold = float(variant.delta_input_threshold) * scale
        hidden_threshold = float(variant.delta_hidden_threshold) * scale
        h1 = runner1.step(x[t], input_threshold, hidden_threshold)

        if t % int(model.skip_len) == 0:
            h2 = runner2.step(h1, input_threshold, hidden_threshold)
            stats.add_dense_matvec(
                readout_weight.shape[0],
                readout_weight.shape[1],
                activation_bits=spec_bits(qspec(variant, "h")),
                weight_bits=spec_bits(qspec(variant, "readout")),
            )
            logits_t = F.linear(fixed_quantize(h2, qspec(variant, "h")), readout_weight, readout_bias)
            logits_t = fixed_quantize(logits_t, qspec(variant, "readout"))
            logits_down.append(logits_t)
            start_prob = float(torch.sigmoid(logits_t[-1]).item())
            if start_prob > float(variant.blank_start_threshold):
                pending_events.append(t + max(1, delay))

        if active_count > 0:
            active_count -= 1

    stats.add_blank_active_ratio(active_steps / max(1, int(length)))
    if not logits_down:
        return torch.zeros((0, model.readout.out_features), device=device)
    logits_down_tensor = torch.stack(logits_down, dim=0)
    return torch.repeat_interleave(logits_down_tensor, repeats=model.skip_len, dim=0)[:length]


def manual_forward_one(
    model: PaperBrainToTextV5,
    inputs: torch.Tensor,
    length: int,
    session_id: int,
    variant: VariantConfig,
    device: torch.device,
    baseline_logits: Optional[torch.Tensor] = None,
    char_start_target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, HardwareStats]:
    stats = HardwareStats()
    lut = (
        LutActivations(size=variant.lut_size, value_range=variant.lut_range, device=device)
        if variant.activation_mode == "lut"
        else None
    )
    x = inputs[:length].to(device)
    session_tensor = torch.tensor([session_id], dtype=torch.long, device=device)

    with torch.no_grad():
        weights = model.session_weight[session_tensor][0]
        bias = model.session_bias[session_tensor][0]
        stats.add_dense_matvec_batch(
            length,
            weights.shape[0],
            weights.shape[1],
            activation_bits=spec_bits(qspec(variant, "input")),
            weight_bits=spec_bits(variant.fixed_point),
        )
        x = x @ weights.T + bias
        x = F.conv1d(
            x.unsqueeze(0).transpose(1, 2),
            model.smoother.kernel.to(device),
            padding=model.smoother.padding,
            groups=model.smoother.channels,
        ).transpose(1, 2).squeeze(0)
        kernel_width = int(model.smoother.kernel.shape[-1])
        stats.add_dense_matvec_batch(
            length,
            int(model.smoother.channels),
            kernel_width,
            activation_bits=spec_bits(qspec(variant, "input")),
            weight_bits=spec_bits(variant.fixed_point),
        )
        x = fixed_quantize(x, qspec(variant, "input"))
        if variant.blank_aware and variant.blank_source == "causal_self":
            logits = causal_blank_gru_forward(
                model=model,
                x=x,
                length=length,
                variant=variant,
                lut=lut,
                stats=stats,
                device=device,
            )
            return logits.detach().cpu(), stats

        input_schedule, hidden_schedule, active_ratio = make_threshold_schedules(
            length=length,
            variant=variant,
            device=device,
            baseline_logits=baseline_logits,
            char_start_target=char_start_target,
        )
        if variant.blank_aware:
            stats.add_blank_active_ratio(active_ratio)

        h0_1 = model.start_state_1[0, 0].to(device)
        gru_runner = delta_gru_sequence if variant.use_delta_gru else dense_gru_sequence
        gru_kwargs = {
            "x_seq": x,
            "weight_ih": model.gru1.weight_ih_l0.to(device),
            "weight_hh": model.gru1.weight_hh_l0.to(device),
            "bias_ih": model.gru1.bias_ih_l0.to(device),
            "bias_hh": model.gru1.bias_hh_l0.to(device),
            "h0": h0_1,
            "variant": variant,
            "lut": lut,
            "stats": stats,
        }
        if variant.use_delta_gru:
            gru_kwargs["input_threshold_schedule"] = input_schedule
            gru_kwargs["hidden_threshold_schedule"] = hidden_schedule
        out1 = gru_runner(**gru_kwargs)
        downsampled = out1[:: model.skip_len]
        down_length = max(1, int(math.floor((length + model.skip_len - 1) / model.skip_len)))
        downsampled = downsampled[:down_length]
        down_input_schedule = downsample_threshold_schedule(input_schedule, model.skip_len, down_length)
        down_hidden_schedule = downsample_threshold_schedule(hidden_schedule, model.skip_len, down_length)

        h0_2 = model.start_state_2[0, 0].to(device)
        gru_kwargs = {
            "x_seq": downsampled,
            "weight_ih": model.gru2.weight_ih_l0.to(device),
            "weight_hh": model.gru2.weight_hh_l0.to(device),
            "bias_ih": model.gru2.bias_ih_l0.to(device),
            "bias_hh": model.gru2.bias_hh_l0.to(device),
            "h0": h0_2,
            "variant": variant,
            "lut": lut,
            "stats": stats,
        }
        if variant.use_delta_gru:
            gru_kwargs["input_threshold_schedule"] = down_input_schedule
            gru_kwargs["hidden_threshold_schedule"] = down_hidden_schedule
        out2 = gru_runner(**gru_kwargs)

        readout_weight = fixed_quantize(model.readout.weight.to(device), qspec(variant, "readout"))
        readout_bias = fixed_quantize(model.readout.bias.to(device), qspec(variant, "readout"))
        stats.add_dense_matvec_batch(
            int(out2.shape[0]),
            readout_weight.shape[0],
            readout_weight.shape[1],
            activation_bits=spec_bits(qspec(variant, "h")),
            weight_bits=spec_bits(qspec(variant, "readout")),
        )
        logits_down = F.linear(fixed_quantize(out2, qspec(variant, "h")), readout_weight, readout_bias)
        logits_down = fixed_quantize(logits_down, qspec(variant, "readout"))
        logits = torch.repeat_interleave(logits_down, repeats=model.skip_len, dim=0)[:length]
    return logits.detach().cpu(), stats


def estimate_dense_forward_stats(model: PaperBrainToTextV5, length: int) -> HardwareStats:
    stats = HardwareStats()
    length = int(length)

    session_rows = int(model.session_weight.shape[1])
    session_cols = int(model.session_weight.shape[2])
    stats.add_dense_matvec_batch(length, session_rows, session_cols)

    kernel_width = int(model.smoother.kernel.shape[-1])
    stats.add_dense_matvec_batch(length, int(model.smoother.channels), kernel_width)

    for _ in range(length):
        stats.add_dense_matvec(model.gru1.weight_ih_l0.shape[0], model.gru1.weight_ih_l0.shape[1])
        stats.add_dense_matvec(model.gru1.weight_hh_l0.shape[0], model.gru1.weight_hh_l0.shape[1])

    down_length = max(1, int(math.floor((length + model.skip_len - 1) / model.skip_len)))
    for _ in range(down_length):
        stats.add_dense_matvec(model.gru2.weight_ih_l0.shape[0], model.gru2.weight_ih_l0.shape[1])
        stats.add_dense_matvec(model.gru2.weight_hh_l0.shape[0], model.gru2.weight_hh_l0.shape[1])

    stats.add_dense_matvec_batch(down_length, int(model.readout.weight.shape[0]), int(model.readout.weight.shape[1]))

    hidden_size = int(model.gru1.hidden_size)
    stats.sigmoid_calls += 2 * hidden_size * (length + down_length)
    stats.tanh_calls += hidden_size * (length + down_length)
    return stats


def load_eval_loader(
    checkpoint: Dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    data_config = checkpoint.get("data_config", {})
    datasets_root = resolve_existing_path(
        str(data_config.get("datasets_root")) if data_config.get("datasets_root") else None,
        args.data_root,
    )
    training_steps_root = resolve_existing_path(
        args.training_steps_root,
        str(data_config.get("training_steps_root")) if data_config.get("training_steps_root") else None,
    )
    if not training_steps_root or not os.path.exists(str(training_steps_root)):
        training_steps_root = find_default_training_steps_root(Path(__file__).resolve().parent)
    if args.use_existing_label_cache and os.path.exists(args.label_cache):
        label_payload = load_existing_label_cache(args.label_cache)
    else:
        label_payload = prepare_paper_v5_label_cache(
            data_root=str(datasets_root),
            cache_path=args.label_cache,
            max_records_per_session=args.max_records_per_session,
            training_steps_root=training_steps_root,
            cv_part=str(data_config.get("cv_part", args.cv_part)),
        )
    cached_datasets_root = str(label_payload.get("datasets_root", datasets_root))
    datasets_root = cached_datasets_root if os.path.exists(cached_datasets_root) else str(datasets_root)
    cached_training_steps_root = label_payload.get("training_steps_root")
    resolved_training_steps_root = (
        cached_training_steps_root
        if cached_training_steps_root and os.path.exists(str(cached_training_steps_root))
        else training_steps_root
    )
    resolved_cv_part = str(label_payload.get("cv_part", args.cv_part))
    records = [record for record in label_payload["records"] if record.split == args.split]
    if len(records) == 0:
        records = [record for record in label_payload["records"] if record.split == "train"][:10]
    dataset = PaperV5EvalDataset(
        data_root=datasets_root,
        records=records,
        bin_size=int(checkpoint["train_config"]["bin_size"]),
        training_steps_root=resolved_training_steps_root,
        cv_part=resolved_cv_part,
    )
    if args.max_samples is not None and int(args.max_samples) > 0:
        dataset = Subset(dataset, list(range(min(int(args.max_samples), len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_variable_paper_batch,
    )


def mixed_precision_kwargs(profile: str) -> Dict[str, object]:
    profile = profile.strip().lower()
    if profile == "balanced":
        return {
            "fixed_point": FixedPointSpec(bits=16, frac_bits=8, enabled=True),
            "input_spec": FixedPointSpec(bits=16, frac_bits=8, enabled=True),
            "weight_spec": FixedPointSpec(bits=16, frac_bits=8, enabled=True),
            "preact_spec": FixedPointSpec(bits=24, frac_bits=10, enabled=True),
            "r_spec": FixedPointSpec(bits=8, frac_bits=7, enabled=True),
            "z_spec": FixedPointSpec(bits=8, frac_bits=7, enabled=True),
            "n_spec": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "h_spec": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "readout_spec": FixedPointSpec(bits=16, frac_bits=8, enabled=True),
            "quant_profile": "mixed_balanced",
        }
    if profile == "aggressive":
        return {
            "fixed_point": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "input_spec": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "weight_spec": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "preact_spec": FixedPointSpec(bits=20, frac_bits=8, enabled=True),
            "r_spec": FixedPointSpec(bits=6, frac_bits=5, enabled=True),
            "z_spec": FixedPointSpec(bits=6, frac_bits=5, enabled=True),
            "n_spec": FixedPointSpec(bits=10, frac_bits=7, enabled=True),
            "h_spec": FixedPointSpec(bits=10, frac_bits=7, enabled=True),
            "readout_spec": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "quant_profile": "mixed_aggressive",
        }
    if profile == "h8":
        return {
            "fixed_point": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "input_spec": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "weight_spec": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "preact_spec": FixedPointSpec(bits=20, frac_bits=8, enabled=True),
            "r_spec": FixedPointSpec(bits=8, frac_bits=7, enabled=True),
            "z_spec": FixedPointSpec(bits=8, frac_bits=7, enabled=True),
            "n_spec": FixedPointSpec(bits=8, frac_bits=6, enabled=True),
            "h_spec": FixedPointSpec(bits=8, frac_bits=6, enabled=True),
            "readout_spec": FixedPointSpec(bits=12, frac_bits=8, enabled=True),
            "quant_profile": "mixed_h8",
        }
    raise ValueError(f"Unsupported mixed precision profile: {profile}")


def gate_sensitivity_spec(gate: str, bits: int) -> FixedPointSpec:
    bits = int(bits)
    if gate in {"r", "z"}:
        frac_bits = max(1, bits - 1)
    elif gate in {"n", "h"}:
        frac_bits = max(1, min(8, bits - 2))
    elif gate == "readout":
        frac_bits = max(1, min(8, bits - 4))
    else:
        raise ValueError(f"Unsupported gate for sensitivity: {gate}")
    return FixedPointSpec(bits=bits, frac_bits=frac_bits, enabled=True)


def build_variants(args: argparse.Namespace) -> List[VariantConfig]:
    q8_spec = FixedPointSpec(bits=16, frac_bits=8, enabled=True)
    dense_float = VariantConfig(name="dense_float", quant_profile="float")
    dense_q8_lut = VariantConfig(
        name="dense_q8_8_lut",
        fixed_point=q8_spec,
        activation_mode="lut",
        lut_size=args.lut_size,
        lut_range=args.lut_range,
        quant_profile="q8_8",
    )
    variants = [dense_float, dense_q8_lut]

    mixed_profiles = [profile.strip() for profile in args.mixed_precision_profiles.split(",") if profile.strip()]
    mixed_kwargs = {profile: mixed_precision_kwargs(profile) for profile in mixed_profiles}
    if args.enable_mixed_precision:
        for profile, kwargs in mixed_kwargs.items():
            variants.append(
                VariantConfig(
                    name=f"dense_mixed_{profile}_lut",
                    activation_mode="lut",
                    lut_size=args.lut_size,
                    lut_range=args.lut_range,
                    **kwargs,
                )
            )

    for x_thr in parse_float_list(args.delta_input_thresholds):
        for h_thr in parse_float_list(args.delta_hidden_thresholds):
            variants.append(
                VariantConfig(
                    name=f"delta_q8_8_lut_x{x_thr:g}_h{h_thr:g}",
                    fixed_point=q8_spec,
                    activation_mode="lut",
                    lut_size=args.lut_size,
                    lut_range=args.lut_range,
                    delta_input_threshold=x_thr,
                    delta_hidden_threshold=h_thr,
                    use_delta_gru=True,
                    quant_profile="q8_8",
                )
            )
            if args.enable_mixed_precision:
                for profile, kwargs in mixed_kwargs.items():
                    variants.append(
                        VariantConfig(
                            name=f"delta_mixed_{profile}_lut_x{x_thr:g}_h{h_thr:g}",
                            activation_mode="lut",
                            lut_size=args.lut_size,
                            lut_range=args.lut_range,
                            delta_input_threshold=x_thr,
                            delta_hidden_threshold=h_thr,
                            use_delta_gru=True,
                            **kwargs,
                        )
            )
            if args.enable_blank_aware:
                blank_prefix = "causal_blank" if args.blank_source == "causal_self" else "blank"
                variants.append(
                    VariantConfig(
                        name=f"{blank_prefix}_delta_q8_8_lut_x{x_thr:g}_h{h_thr:g}",
                        fixed_point=q8_spec,
                        activation_mode="lut",
                        lut_size=args.lut_size,
                        lut_range=args.lut_range,
                        delta_input_threshold=x_thr,
                        delta_hidden_threshold=h_thr,
                        use_delta_gru=True,
                        blank_aware=True,
                        blank_source=args.blank_source,
                        blank_start_threshold=float(args.blank_start_threshold),
                        blank_context_radius=int(args.blank_context_radius),
                        blank_threshold_scale=float(args.blank_threshold_scale),
                        char_start_threshold_scale=float(args.char_start_threshold_scale),
                        causal_control_delay=int(args.causal_control_delay),
                        quant_profile="q8_8",
                    )
                )
                if args.enable_mixed_precision:
                    for profile, kwargs in mixed_kwargs.items():
                        variants.append(
                            VariantConfig(
                                name=f"{blank_prefix}_delta_mixed_{profile}_lut_x{x_thr:g}_h{h_thr:g}",
                                activation_mode="lut",
                                lut_size=args.lut_size,
                                lut_range=args.lut_range,
                                delta_input_threshold=x_thr,
                                delta_hidden_threshold=h_thr,
                                use_delta_gru=True,
                                blank_aware=True,
                                blank_source=args.blank_source,
                                blank_start_threshold=float(args.blank_start_threshold),
                                blank_context_radius=int(args.blank_context_radius),
                                blank_threshold_scale=float(args.blank_threshold_scale),
                                char_start_threshold_scale=float(args.char_start_threshold_scale),
                                causal_control_delay=int(args.causal_control_delay),
                                **kwargs,
                            )
                        )

    if args.enable_gate_sensitivity:
        base_profile = args.gate_sensitivity_base_profile.strip()
        base_kwargs = mixed_precision_kwargs(base_profile)
        gates = [gate.strip() for gate in args.gate_sensitivity_gates.split(",") if gate.strip()]
        bits_list = [int(bit.strip()) for bit in args.gate_sensitivity_bits.split(",") if bit.strip()]
        for x_thr in parse_float_list(args.delta_input_thresholds):
            for h_thr in parse_float_list(args.delta_hidden_thresholds):
                for gate in gates:
                    for bits in bits_list:
                        kwargs = dict(base_kwargs)
                        kwargs[f"{gate}_spec"] = gate_sensitivity_spec(gate, bits)
                        kwargs["quant_profile"] = f"gate_sense_{base_profile}_{gate}{bits}"
                        variants.append(
                            VariantConfig(
                                name=f"delta_gate_sense_{base_profile}_{gate}{bits}_x{x_thr:g}_h{h_thr:g}",
                                activation_mode="lut",
                                lut_size=args.lut_size,
                                lut_range=args.lut_range,
                                delta_input_threshold=x_thr,
                                delta_hidden_threshold=h_thr,
                                use_delta_gru=True,
                                **kwargs,
                            )
                        )

    if args.variant_regex:
        pattern = re.compile(args.variant_regex)
        variants = [variant for variant in variants if pattern.search(variant.name)]
    return variants


def analyze_session_affine(model: PaperBrainToTextV5, ranks: Sequence[int]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        weights = model.session_weight.detach().cpu()
        biases = model.session_bias.detach().cpu()
        input_size = weights.shape[-1]
        eye = torch.eye(input_size)
        dense_params = input_size * input_size + input_size
        for session_id in range(weights.shape[0]):
            delta = weights[session_id] - eye
            fro = torch.linalg.norm(delta).item()
            singular_values = torch.linalg.svdvals(delta)
            total_energy = float(torch.sum(singular_values ** 2).item())
            for rank in ranks:
                rank = min(int(rank), input_size)
                tail_energy = float(torch.sum(singular_values[rank:] ** 2).item())
                rel_error = math.sqrt(tail_energy / total_energy) if total_energy > 0 else 0.0
                low_rank_params = input_size + 2 * input_size * rank + input_size
                rows.append(
                    {
                        "session_id": float(session_id),
                        "rank": float(rank),
                        "delta_fro_norm": float(fro),
                        "relative_fro_error": float(rel_error),
                        "dense_params": float(dense_params),
                        "diag_low_rank_params": float(low_rank_params),
                        "param_reduction": float(1.0 - low_rank_params / dense_params),
                        "bias_l2_norm": float(torch.linalg.norm(biases[session_id]).item()),
                    }
                )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def sample_error_counts(reference: str, hypothesis: str) -> Dict[str, object]:
    reference_display = render_transcript(reference)
    hypothesis_display = render_transcript(hypothesis)
    reference_words = reference_display.split()
    hypothesis_words = hypothesis_display.split()
    char_errors = edit_distance(list(reference_display), list(hypothesis_display))
    char_total = max(1, len(reference_display))
    word_errors = edit_distance(reference_words, hypothesis_words)
    word_total = max(1, len(reference_words))
    return {
        "char_errors": char_errors,
        "char_total": char_total,
        "word_errors": word_errors,
        "word_total": word_total,
        "sample_cer": char_errors / char_total,
        "sample_wer": word_errors / word_total,
    }


def run(args: argparse.Namespace) -> None:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))

    model_device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    manual_device = torch.device("cpu" if args.manual_device == "auto" else args.manual_device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PaperBrainToTextV5(**checkpoint["model_config"]).to(model_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    affine_rows = analyze_session_affine(model, ranks=[int(v) for v in args.affine_ranks.split(",") if v.strip()])
    write_csv(output_dir / "session_affine_low_rank.csv", affine_rows)

    loader = load_eval_loader(checkpoint, args, model_device)
    variants = build_variants(args)
    if not variants:
        raise ValueError("No variants selected. Check --variant_regex.")
    print(f"Model device: {model_device}; manual GRU device: {manual_device}; variants: {len(variants)}", flush=True)
    for variant in variants:
        print(f"  - {variant.name}", flush=True)
    results: List[Dict[str, object]] = []
    example_rows: List[Dict[str, object]] = []
    baseline_logits: List[torch.Tensor] = []
    references: List[str] = []
    lengths_cache: List[int] = []
    input_cache: List[torch.Tensor] = []
    session_cache: List[int] = []
    metadata_cache: List[Dict[str, object]] = []
    char_start_cache: List[torch.Tensor] = []
    per_sample_path = output_dir / "hardware_algorithm_per_sample.csv"
    if per_sample_path.exists():
        per_sample_path.unlink()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            inputs = batch["inputs"].to(model_device)
            lengths = batch["input_lengths"].to(model_device)
            session_ids = batch["session_ids"].to(model_device)
            char_starts = batch["char_start_target"]
            if args.truncate_time_steps is not None:
                trunc = int(args.truncate_time_steps)
                lengths = torch.clamp(lengths, max=trunc)
                inputs = inputs[:, : int(lengths.max().item()), :]
                char_starts = char_starts[:, : int(lengths.max().item())]
            logits, _ = model(inputs, lengths, session_ids)
            for row in range(inputs.shape[0]):
                length = int(lengths[row].item())
                references.append(batch["transcripts"][row])
                lengths_cache.append(length)
                input_cache.append(inputs[row, :length].detach().cpu())
                session_id = int(session_ids[row].item())
                session_cache.append(session_id)
                metadata_cache.append(
                    {
                        "sample_index": len(references),
                        "session_id": session_id,
                        "session_name": batch["session_names"][row],
                        "sentence_index": int(batch["sentence_indices"][row].item()),
                        "block_num": int(batch["block_nums"][row].item()),
                        "input_length": length,
                    }
                )
                char_start_cache.append(char_starts[row, :length].detach().cpu())
                baseline_logits.append(logits[row, :length].detach().cpu())
            print(f"Cached baseline batch {batch_idx}/{len(loader)}", flush=True)

    model = model.to(manual_device)
    model.eval()

    for variant in variants:
        hypotheses: List[str] = []
        aggregate_stats = HardwareStats()
        mse_values: List[float] = []
        max_abs_values: List[float] = []
        sample_rows: List[Dict[str, object]] = []
        for idx, (inputs, length, session_id, metadata, baseline, char_start) in enumerate(
            zip(input_cache, lengths_cache, session_cache, metadata_cache, baseline_logits, char_start_cache),
            start=1,
        ):
            if variant.name == "dense_float" and args.reuse_baseline_for_dense_float:
                logits = baseline
                stats = estimate_dense_forward_stats(model, length)
            else:
                logits, stats = manual_forward_one(
                    model,
                    inputs,
                    length,
                    session_id,
                    variant,
                    manual_device,
                    baseline_logits=baseline,
                    char_start_target=char_start,
                )
            aggregate_stats.merge(stats)
            mse_values.append(float(torch.mean((logits - baseline) ** 2).item()))
            max_abs_values.append(float(torch.max(torch.abs(logits - baseline)).item()))
            hyp = decode_paper_v5_outputs(
                logits,
                length,
                start_threshold=float(args.start_threshold),
                start_delay=int(args.start_delay),
            )
            hypotheses.append(hyp)
            sample_stat_summary = stats.summary()
            sample_rows.append(
                {
                    "variant": variant.name,
                    **metadata,
                    "reference": render_transcript(references[idx - 1]),
                    "hypothesis": render_transcript(hyp),
                    **sample_error_counts(references[idx - 1], hyp),
                    "logit_mse_vs_pytorch": mse_values[-1],
                    "logit_max_abs_vs_pytorch": max_abs_values[-1],
                    **sample_stat_summary,
                }
            )
            if idx <= args.max_examples:
                example_rows.append(
                    {
                        "variant": variant.name,
                        **metadata,
                        "reference": render_transcript(references[idx - 1]),
                        "hypothesis": render_transcript(hyp),
                        "logit_mse_vs_pytorch": mse_values[-1],
                        "logit_max_abs_vs_pytorch": max_abs_values[-1],
                    }
                )
            print(f"Variant {variant.name}: sample {idx}/{len(input_cache)}", flush=True)

        metrics = compute_error_rates(references, hypotheses)
        stat_summary = aggregate_stats.summary()
        row: Dict[str, object] = {
            "variant": variant.name,
            "samples": len(references),
            "cer": float(metrics["cer"]),
            "wer": float(metrics["wer"]),
            "mean_logit_mse_vs_pytorch": float(sum(mse_values) / max(1, len(mse_values))),
            "max_logit_abs_vs_pytorch": float(max(max_abs_values) if max_abs_values else 0.0),
            "quant_profile": variant.quant_profile,
            "quant_specs": quant_profile_summary(variant),
            "fixed_point": variant.fixed_point.enabled,
            "bits": variant.fixed_point.bits,
            "frac_bits": variant.fixed_point.frac_bits,
            "input_bits": qspec(variant, "input").bits if qspec(variant, "input").enabled else 0,
            "weight_bits": qspec(variant, "weight").bits if qspec(variant, "weight").enabled else 0,
            "preact_bits": qspec(variant, "preact").bits if qspec(variant, "preact").enabled else 0,
            "r_bits": qspec(variant, "r").bits if qspec(variant, "r").enabled else 0,
            "z_bits": qspec(variant, "z").bits if qspec(variant, "z").enabled else 0,
            "n_bits": qspec(variant, "n").bits if qspec(variant, "n").enabled else 0,
            "h_bits": qspec(variant, "h").bits if qspec(variant, "h").enabled else 0,
            "readout_bits": qspec(variant, "readout").bits if qspec(variant, "readout").enabled else 0,
            "activation_mode": variant.activation_mode,
            "lut_size": variant.lut_size,
            "delta_input_threshold": variant.delta_input_threshold,
            "delta_hidden_threshold": variant.delta_hidden_threshold,
            "use_delta_gru": variant.use_delta_gru,
            "blank_aware": variant.blank_aware,
            "blank_source": variant.blank_source,
            "blank_start_threshold": variant.blank_start_threshold,
            "blank_context_radius": variant.blank_context_radius,
            "blank_threshold_scale": variant.blank_threshold_scale,
            "char_start_threshold_scale": variant.char_start_threshold_scale,
            "causal_control_delay": variant.causal_control_delay,
        }
        row.update(stat_summary)
        results.append(row)
        append_csv(per_sample_path, sample_rows)
        write_csv(output_dir / "hardware_algorithm_results.csv", results)
        write_csv(output_dir / "hardware_algorithm_examples.csv", example_rows)

    write_csv(output_dir / "hardware_algorithm_results.csv", results)
    write_csv(output_dir / "hardware_algorithm_examples.csv", example_rows)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hardware-friendly GRU ideas for BCI brain-to-text decoding.")
    parser.add_argument("--checkpoint", type=str, default="brain_to_text_v5_paper_best.pt")
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "Datasets"))
    parser.add_argument("--training_steps_root", type=str, default=None)
    parser.add_argument("--cv_part", type=str, default="HeldOutBlocks", choices=["HeldOutBlocks", "HeldOutTrials"])
    parser.add_argument("--label_cache", type=str, default="paper_v5_label_cache.pt")
    parser.add_argument("--use_existing_label_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_records_per_session", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--manual_device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--truncate_time_steps", type=int, default=None)
    parser.add_argument("--start_threshold", type=float, default=0.3)
    parser.add_argument("--start_delay", type=int, default=15)
    parser.add_argument("--delta_input_thresholds", type=str, default="0.02")
    parser.add_argument("--delta_hidden_thresholds", type=str, default="0.01")
    parser.add_argument("--enable_mixed_precision", action="store_true")
    parser.add_argument("--mixed_precision_profiles", type=str, default="balanced,aggressive,h8")
    parser.add_argument("--enable_gate_sensitivity", action="store_true")
    parser.add_argument("--gate_sensitivity_base_profile", type=str, default="balanced")
    parser.add_argument("--gate_sensitivity_gates", type=str, default="r,z,n,h,readout")
    parser.add_argument("--gate_sensitivity_bits", type=str, default="6,8,10,12,16")
    parser.add_argument("--enable_blank_aware", action="store_true")
    parser.add_argument("--blank_source", type=str, default="baseline_start", choices=["baseline_start", "target", "causal_self"])
    parser.add_argument("--blank_start_threshold", type=float, default=0.3)
    parser.add_argument("--blank_context_radius", type=int, default=15)
    parser.add_argument("--blank_threshold_scale", type=float, default=2.0)
    parser.add_argument("--char_start_threshold_scale", type=float, default=0.5)
    parser.add_argument("--causal_control_delay", type=int, default=1)
    parser.add_argument("--variant_regex", type=str, default=None)
    parser.add_argument("--reuse_baseline_for_dense_float", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lut_size", type=int, default=1024)
    parser.add_argument("--lut_range", type=float, default=8.0)
    parser.add_argument("--affine_ranks", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--max_examples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="artifacts/hardware_algorithms")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
