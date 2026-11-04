from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from brain_to_text_main_v5 import compute_error_rates, edit_distance, render_transcript
from evaluate_bci_gru_hardware_algorithms import (
    FixedPointSpec,
    HardwareStats,
    LutActivations,
    VariantConfig,
    apply_gate_activation,
    fixed_quantize,
    load_eval_loader,
    mixed_precision_kwargs,
    qspec,
    spec_bits,
    write_csv,
)
from paper_v5_pipeline import PaperBrainToTextV5, decode_paper_v5_outputs

GATE_NAMES = ("r", "z", "n")


@dataclass(frozen=True)
class SchedulerConfig:
    base_input_threshold: float = 0.08
    base_hidden_threshold: float = 0.02
    protect_scale: float = 0.5
    aggressive_scale: float = 2.0
    high_start_threshold: float = 0.3
    blank_threshold: float = 0.08
    low_margin_threshold: float = 1.0
    high_margin_threshold: float = 2.5
    drift_budget: float = 5000.0
    protect_alpha: float = 512.0
    normal_alpha: float = 2048.0
    aggressive_alpha: float = 4096.0
    drift_normalizer: str = "raw"
    enable_semantic: bool = False
    enable_drift_bound: bool = False
    gate_policy: str = "uniform"
    signal_policy: str = "semantic"
    active_budget_policy: str = "none"
    budget_aggressive_ratio: float = 1.0
    budget_normal_ratio: float = 1.0
    budget_protect_ratio: float = 1.0
    precision_policy: str = "none"
    precision_fake_quant: bool = False
    precision_aggressive_bits: int = 16
    precision_normal_bits: int = 16
    precision_protect_bits: int = 16


@dataclass
class StepDiagnostics:
    input_active_ratio: float
    hidden_active_ratio: float
    input_delta_norm: float
    hidden_delta_norm: float
    input_residual_bound: float
    hidden_residual_bound: float
    input_raw_residual_bound: float
    hidden_raw_residual_bound: float
    force_refresh: bool
    input_gate_active_ratios: Dict[str, float] = None
    hidden_gate_active_ratios: Dict[str, float] = None
    input_gate_residual_bounds: Dict[str, float] = None
    hidden_gate_residual_bounds: Dict[str, float] = None
    input_gate_raw_residual_bounds: Dict[str, float] = None
    hidden_gate_raw_residual_bounds: Dict[str, float] = None
    input_gate_active_indices: Dict[str, List[int]] = None
    hidden_gate_active_indices: Dict[str, List[int]] = None
    dense_weight_bytes: float = 0.0
    actual_weight_bytes: float = 0.0
    input_activation_bytes: float = 0.0
    hidden_activation_bytes: float = 0.0
    reference_write_bytes: float = 0.0
    hidden_state_write_bytes: float = 0.0
    metadata_bytes: float = 0.0
    total_traffic_bytes: float = 0.0

    def __post_init__(self) -> None:
        for name in [
            "input_gate_active_ratios",
            "hidden_gate_active_ratios",
            "input_gate_residual_bounds",
            "hidden_gate_residual_bounds",
            "input_gate_raw_residual_bounds",
            "hidden_gate_raw_residual_bounds",
            "input_gate_active_indices",
            "hidden_gate_active_indices",
        ]:
            if getattr(self, name) is None:
                if name.endswith("_active_indices"):
                    setattr(self, name, {gate: [] for gate in GATE_NAMES})
                else:
                    setattr(self, name, {gate: 0.0 for gate in GATE_NAMES})


def top_margin(logits: torch.Tensor) -> float:
    char_logits = logits[:-1]
    if char_logits.numel() < 2:
        return 0.0
    values = torch.topk(char_logits, k=2).values
    return float((values[0] - values[1]).item())


def mode_thresholds(mode: str, config: SchedulerConfig) -> Tuple[float, float]:
    scale = 1.0
    if mode == "PROTECT":
        scale = float(config.protect_scale)
    elif mode == "AGGRESSIVE":
        scale = float(config.aggressive_scale)
    return (
        float(config.base_input_threshold) * scale,
        float(config.base_hidden_threshold) * scale,
    )


def mode_budget_ratio(mode: str, config: SchedulerConfig) -> Optional[float]:
    policy = str(config.active_budget_policy)
    if policy == "none":
        return None
    if policy not in {"uniform", "semantic"}:
        raise ValueError(f"Unsupported active_budget_policy: {policy}")
    if policy == "uniform":
        return float(config.budget_normal_ratio)
    if mode == "PROTECT":
        return float(config.budget_protect_ratio)
    if mode == "AGGRESSIVE":
        return float(config.budget_aggressive_ratio)
    return float(config.budget_normal_ratio)


def mode_precision_bits(mode: str, config: SchedulerConfig) -> Optional[int]:
    policy = str(config.precision_policy)
    if policy == "none":
        return None
    if policy != "semantic":
        raise ValueError(f"Unsupported precision_policy: {policy}")
    if mode == "PROTECT":
        return int(config.precision_protect_bits)
    if mode == "AGGRESSIVE":
        return int(config.precision_aggressive_bits)
    return int(config.precision_normal_bits)


def phase_qspec(base_spec: FixedPointSpec, role: str, bits: Optional[int]) -> FixedPointSpec:
    if bits is None or not base_spec.enabled:
        return base_spec
    if int(bits) >= 16:
        return base_spec
    capped_bits = max(2, min(int(bits), int(base_spec.bits)))
    if capped_bits >= int(base_spec.bits):
        return base_spec
    role = str(role)
    if role in {"r", "z"}:
        frac_bits = max(1, capped_bits - 1)
    elif role in {"n", "h"}:
        frac_bits = max(1, min(int(base_spec.frac_bits), capped_bits - 2))
    elif role == "preact":
        frac_bits = max(1, min(int(base_spec.frac_bits), capped_bits - 4))
    else:
        frac_bits = max(1, min(int(base_spec.frac_bits), capped_bits - 4))
    return FixedPointSpec(bits=capped_bits, frac_bits=frac_bits, enabled=True)


def phase_fake_quantize(
    x: torch.Tensor,
    variant: VariantConfig,
    role: str,
    bits: Optional[int],
    enabled: bool,
) -> torch.Tensor:
    base_spec = qspec(variant, role)
    if not enabled:
        return fixed_quantize(x, base_spec)
    return fixed_quantize(x, phase_qspec(base_spec, role, bits))


def add_delta_matvec_with_effective_bits(
    stats: HardwareStats,
    rows: int,
    cols: int,
    active_cols: int,
    *,
    input_side: bool,
    dense_activation_bits: int,
    dense_weight_bits: int,
    actual_activation_bits: int,
    actual_weight_bits: int,
) -> None:
    rows = int(rows)
    cols = int(cols)
    active_cols = int(active_cols)
    stats.add_delta_matvec(
        rows,
        cols,
        active_cols,
        input_side=input_side,
        activation_bits=int(dense_activation_bits),
        weight_bits=int(dense_weight_bits),
    )
    actual_macs = rows * active_cols
    stats.actual_bops += actual_macs * (
        int(actual_activation_bits) * int(actual_weight_bits)
        - int(dense_activation_bits) * int(dense_weight_bits)
    )
    stats.actual_weight_read_bits += actual_macs * (int(actual_weight_bits) - int(dense_weight_bits))


def add_dense_matvec_with_effective_bits(
    stats: HardwareStats,
    rows: int,
    cols: int,
    *,
    dense_activation_bits: int,
    dense_weight_bits: int,
    actual_activation_bits: int,
    actual_weight_bits: int,
) -> None:
    rows = int(rows)
    cols = int(cols)
    stats.add_dense_matvec(
        rows,
        cols,
        activation_bits=int(dense_activation_bits),
        weight_bits=int(dense_weight_bits),
    )
    macs = rows * cols
    stats.actual_bops += macs * (
        int(actual_activation_bits) * int(actual_weight_bits)
        - int(dense_activation_bits) * int(dense_weight_bits)
    )
    stats.actual_weight_read_bits += macs * (int(actual_weight_bits) - int(dense_weight_bits))


def gate_mode_thresholds(
    mode: str,
    config: SchedulerConfig,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    if str(config.gate_policy) == "uniform":
        return None, None
    input_threshold, hidden_threshold = mode_thresholds(mode, config)
    scales = {gate: 1.0 for gate in GATE_NAMES}
    policy = str(config.gate_policy)
    if policy == "semantic_zn":
        if mode == "PROTECT":
            scales.update({"r": 1.0, "z": 0.75, "n": 0.75})
        elif mode == "AGGRESSIVE":
            scales.update({"r": 1.5, "z": 0.75, "n": 1.25})
    elif policy == "reset_relaxed":
        if mode == "AGGRESSIVE":
            scales.update({"r": 2.0, "z": 1.0, "n": 1.0})
    elif policy == "candidate_relaxed":
        if mode == "AGGRESSIVE":
            scales.update({"r": 1.0, "z": 1.0, "n": 2.0})
    elif policy == "all_relaxed":
        if mode == "AGGRESSIVE":
            scales.update({"r": 1.5, "z": 1.25, "n": 1.5})
        elif mode == "PROTECT":
            scales.update({"r": 0.9, "z": 0.75, "n": 0.75})
    elif policy == "z_protected_n_relaxed":
        if mode == "AGGRESSIVE":
            scales.update({"r": 1.25, "z": 0.75, "n": 1.75})
        elif mode == "PROTECT":
            scales.update({"r": 1.0, "z": 0.65, "n": 0.85})
    elif policy == "rn_relaxed_z_protected":
        if mode == "AGGRESSIVE":
            scales.update({"r": 1.75, "z": 0.75, "n": 1.75})
        elif mode == "PROTECT":
            scales.update({"r": 1.0, "z": 0.65, "n": 1.0})
    elif policy == "all_mild":
        if mode == "AGGRESSIVE":
            scales.update({"r": 1.25, "z": 1.1, "n": 1.25})
        elif mode == "PROTECT":
            scales.update({"r": 0.9, "z": 0.85, "n": 0.9})
    elif policy == "aggressive_only":
        if mode == "AGGRESSIVE":
            scales.update({"r": 1.5, "z": 0.9, "n": 1.5})
    else:
        raise ValueError(f"Unsupported gate_policy: {policy}")
    return (
        {gate: input_threshold * scales[gate] for gate in GATE_NAMES},
        {gate: hidden_threshold * scales[gate] for gate in GATE_NAMES},
    )


def mode_alpha(mode: str, config: SchedulerConfig) -> float:
    if mode == "PROTECT":
        return float(config.protect_alpha)
    if mode == "AGGRESSIVE":
        return float(config.aggressive_alpha)
    return float(config.normal_alpha)


def diagnostic_trace_fields(prefix: str, diag: StepDiagnostics, include: bool = True) -> Dict[str, object]:
    if not include:
        empty: Dict[str, object] = {
            f"{prefix}_dense_weight_bytes": "",
            f"{prefix}_actual_weight_bytes": "",
            f"{prefix}_input_activation_bytes": "",
            f"{prefix}_hidden_activation_bytes": "",
            f"{prefix}_reference_write_bytes": "",
            f"{prefix}_hidden_state_write_bytes": "",
            f"{prefix}_metadata_bytes": "",
            f"{prefix}_total_traffic_bytes": "",
        }
        for side in ("input", "hidden"):
            for gate in GATE_NAMES:
                empty[f"{prefix}_{side}_{gate}_active_ratio"] = ""
                empty[f"{prefix}_{side}_{gate}_residual_bound"] = ""
                empty[f"{prefix}_{side}_{gate}_raw_residual_bound"] = ""
        return empty
    fields: Dict[str, object] = {
        f"{prefix}_dense_weight_bytes": diag.dense_weight_bytes,
        f"{prefix}_actual_weight_bytes": diag.actual_weight_bytes,
        f"{prefix}_input_activation_bytes": diag.input_activation_bytes,
        f"{prefix}_hidden_activation_bytes": diag.hidden_activation_bytes,
        f"{prefix}_reference_write_bytes": diag.reference_write_bytes,
        f"{prefix}_hidden_state_write_bytes": diag.hidden_state_write_bytes,
        f"{prefix}_metadata_bytes": diag.metadata_bytes,
        f"{prefix}_total_traffic_bytes": diag.total_traffic_bytes,
    }
    for side, active, residual, raw_residual in [
        (
            "input",
            diag.input_gate_active_ratios,
            diag.input_gate_residual_bounds,
            diag.input_gate_raw_residual_bounds,
        ),
        (
            "hidden",
            diag.hidden_gate_active_ratios,
            diag.hidden_gate_residual_bounds,
            diag.hidden_gate_raw_residual_bounds,
        ),
    ]:
        for gate in GATE_NAMES:
            fields[f"{prefix}_{side}_{gate}_active_ratio"] = active.get(gate, 0.0)
            fields[f"{prefix}_{side}_{gate}_residual_bound"] = residual.get(gate, 0.0)
            fields[f"{prefix}_{side}_{gate}_raw_residual_bound"] = raw_residual.get(gate, 0.0)
    return fields


def choose_mode(
    previous_start_prob: float,
    previous_margin: float,
    residual_bound: float,
    config: SchedulerConfig,
) -> str:
    if not config.enable_semantic and not config.enable_drift_bound:
        return "NORMAL"
    if config.enable_drift_bound and residual_bound > float(config.drift_budget):
        return "REFRESH"
    if config.enable_semantic:
        policy = str(config.signal_policy)
        if policy == "blank_only":
            if previous_start_prob > float(config.high_start_threshold):
                return "PROTECT"
            if previous_start_prob < float(config.blank_threshold):
                return "AGGRESSIVE"
            return "NORMAL"
        if policy == "margin_only":
            if previous_margin < float(config.low_margin_threshold):
                return "PROTECT"
            if previous_margin > float(config.high_margin_threshold):
                return "AGGRESSIVE"
            return "NORMAL"
        if policy not in {"semantic", "aggressive_fsm"}:
            raise ValueError(f"Unsupported signal_policy: {policy}")
        if previous_start_prob > float(config.high_start_threshold):
            return "PROTECT"
        if previous_margin < float(config.low_margin_threshold):
            return "PROTECT"
        if previous_start_prob < float(config.blank_threshold) and previous_margin > float(config.high_margin_threshold):
            return "AGGRESSIVE"
    return "NORMAL"


class InstrumentedGruRuntime:
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
        drift_normalizer: str = "raw",
    ) -> None:
        self.weight_ih = fixed_quantize(weight_ih, qspec(variant, "weight"))
        self.weight_hh = fixed_quantize(weight_hh, qspec(variant, "weight"))
        self.bias_ih = fixed_quantize(bias_ih, qspec(variant, "preact"))
        self.bias_hh = fixed_quantize(bias_hh, qspec(variant, "preact"))
        self.h_prev = fixed_quantize(h0, qspec(variant, "h"))
        self.variant = variant
        self.lut = lut
        self.stats = stats
        self.drift_normalizer = drift_normalizer
        self.x_ref: Optional[torch.Tensor] = None
        self.h_ref = torch.zeros_like(self.h_prev)
        self.x_ref_gate: Optional[Dict[str, torch.Tensor]] = None
        self.h_ref_gate: Optional[Dict[str, torch.Tensor]] = None
        self.gi = self.bias_ih.clone()
        self.gh = self.bias_hh.clone()
        self.step_index = 0
        self.input_sensitivity = torch.sum(torch.abs(self.weight_ih), dim=0)
        self.hidden_sensitivity = torch.sum(torch.abs(self.weight_hh), dim=0)
        gate_rows = int(self.weight_ih.shape[0] // 3)
        self.gate_slices = {
            "r": slice(0, gate_rows),
            "z": slice(gate_rows, 2 * gate_rows),
            "n": slice(2 * gate_rows, 3 * gate_rows),
        }
        self.input_gate_sensitivity = {
            gate: torch.sum(torch.abs(self.weight_ih[row_slice, :]), dim=0)
            for gate, row_slice in self.gate_slices.items()
        }
        self.hidden_gate_sensitivity = {
            gate: torch.sum(torch.abs(self.weight_hh[row_slice, :]), dim=0)
            for gate, row_slice in self.gate_slices.items()
        }
        self.last_diag = StepDiagnostics(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    def _normalize_residual_bound(self, raw_bound: float, side: str, residual: torch.Tensor) -> float:
        normalizer = str(self.drift_normalizer)
        if normalizer == "raw":
            return float(raw_bound)
        if side == "input":
            weight = self.weight_ih
            sensitivity = self.input_sensitivity
            preact = self.gi
        else:
            weight = self.weight_hh
            sensitivity = self.hidden_sensitivity
            preact = self.gh
        if normalizer == "per_output":
            return float(raw_bound) / max(1, int(weight.shape[0]))
        if normalizer == "sensitivity_mean":
            denom = float(torch.mean(sensitivity).item()) * max(1, int(residual.numel()))
            return float(raw_bound) / max(denom, 1e-6)
        if normalizer == "preact_norm":
            denom = float(torch.linalg.vector_norm(preact).item())
            return float(raw_bound) / max(denom, 1e-6)
        raise ValueError(f"Unsupported drift_normalizer: {normalizer}")

    def _normalize_gate_residual_bound(
        self,
        raw_bound: float,
        side: str,
        gate: str,
        residual: torch.Tensor,
    ) -> float:
        normalizer = str(self.drift_normalizer)
        if normalizer == "raw":
            return float(raw_bound)
        if side == "input":
            sensitivity = self.input_gate_sensitivity[gate]
            preact = self.gi[self.gate_slices[gate]]
        else:
            sensitivity = self.hidden_gate_sensitivity[gate]
            preact = self.gh[self.gate_slices[gate]]
        if normalizer == "per_output":
            return float(raw_bound) / max(1, int(preact.numel()))
        if normalizer == "sensitivity_mean":
            denom = float(torch.mean(sensitivity).item()) * max(1, int(residual.numel()))
            return float(raw_bound) / max(denom, 1e-6)
        if normalizer == "preact_norm":
            denom = float(torch.linalg.vector_norm(preact).item())
            return float(raw_bound) / max(denom, 1e-6)
        raise ValueError(f"Unsupported drift_normalizer: {normalizer}")

    def _gate_residual_bounds(
        self,
        residual: torch.Tensor,
        side: str,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        sensitivities = self.input_gate_sensitivity if side == "input" else self.hidden_gate_sensitivity
        normalized: Dict[str, float] = {}
        raw: Dict[str, float] = {}
        abs_residual = torch.abs(residual)
        for gate in GATE_NAMES:
            raw_bound = float(torch.sum(abs_residual * sensitivities[gate]).item())
            raw[gate] = raw_bound
            normalized[gate] = self._normalize_gate_residual_bound(raw_bound, side, gate, residual)
        return normalized, raw

    def _traffic_bytes(
        self,
        active_x_count: int,
        active_h_count: int,
        gatewise: bool = False,
        *,
        input_bits_override: Optional[int] = None,
        hidden_bits_override: Optional[int] = None,
        weight_bits_override: Optional[int] = None,
    ) -> Dict[str, float]:
        input_bits = int(input_bits_override) if input_bits_override is not None else spec_bits(qspec(self.variant, "input"))
        hidden_bits = int(hidden_bits_override) if hidden_bits_override is not None else spec_bits(qspec(self.variant, "h"))
        weight_bits = int(weight_bits_override) if weight_bits_override is not None else spec_bits(qspec(self.variant, "weight"))
        input_cols = int(self.weight_ih.shape[1])
        hidden_cols = int(self.weight_hh.shape[1])
        rows = int(self.weight_ih.shape[0])
        gate_rows = int(rows // 3)
        hidden_size = int(self.weight_hh.shape[1])
        dense_weight_bytes = (rows * input_cols + rows * hidden_cols) * weight_bits / 8.0
        active_rows = gate_rows if gatewise else rows
        actual_weight_bytes = active_rows * (int(active_x_count) + int(active_h_count)) * weight_bits / 8.0
        input_activation_bytes = int(active_x_count) * input_bits / 8.0
        hidden_activation_bytes = int(active_h_count) * hidden_bits / 8.0
        reference_write_bytes = input_activation_bytes + hidden_activation_bytes
        hidden_state_write_bytes = hidden_size * hidden_bits / 8.0
        mask_sets = len(GATE_NAMES) if gatewise else 1
        metadata_bytes = mask_sets * (math.ceil(input_cols / 8.0) + math.ceil(hidden_cols / 8.0)) + 1.0
        total_traffic_bytes = (
            actual_weight_bytes
            + input_activation_bytes
            + hidden_activation_bytes
            + reference_write_bytes
            + hidden_state_write_bytes
            + metadata_bytes
        )
        return {
            "dense_weight_bytes": dense_weight_bytes,
            "actual_weight_bytes": actual_weight_bytes,
            "input_activation_bytes": input_activation_bytes,
            "hidden_activation_bytes": hidden_activation_bytes,
            "reference_write_bytes": reference_write_bytes,
            "hidden_state_write_bytes": hidden_state_write_bytes,
            "metadata_bytes": metadata_bytes,
            "total_traffic_bytes": total_traffic_bytes,
        }

    @staticmethod
    def _active_indices(active: torch.Tensor) -> List[int]:
        return torch.argwhere(active.detach()).reshape(-1).to("cpu").tolist()

    @staticmethod
    def _cap_active_by_budget(
        active: torch.Tensor,
        score: torch.Tensor,
        budget_ratio: Optional[float],
    ) -> torch.Tensor:
        if budget_ratio is None:
            return active
        ratio = float(budget_ratio)
        if not math.isfinite(ratio) or ratio >= 1.0:
            return active
        total = int(active.numel())
        if total <= 0:
            return active
        k = max(1, min(total, int(math.ceil(total * max(0.0, ratio)))))
        active_count = int(active.sum().item())
        if active_count <= k:
            return active
        candidate_idx = torch.argwhere(active).reshape(-1)
        if candidate_idx.numel() <= k:
            return active
        candidate_score = score[candidate_idx]
        keep_local = torch.topk(candidate_score, k=k, largest=True).indices
        keep_idx = candidate_idx[keep_local]
        capped = torch.zeros_like(active, dtype=torch.bool)
        capped[keep_idx] = True
        return capped

    def _step_gatewise(
        self,
        x_t: torch.Tensor,
        input_gate_thresholds: Dict[str, float],
        hidden_gate_thresholds: Dict[str, float],
        force_refresh: bool = False,
        residual_alpha: Optional[float] = None,
        margin: float = 0.0,
    ) -> torch.Tensor:
        if self.x_ref_gate is None:
            self.x_ref_gate = {gate: torch.zeros_like(x_t) for gate in GATE_NAMES}
        if self.h_ref_gate is None:
            self.h_ref_gate = {gate: torch.zeros_like(self.h_prev) for gate in GATE_NAMES}

        input_gate_residual_bounds: Dict[str, float] = {}
        hidden_gate_residual_bounds: Dict[str, float] = {}
        input_gate_raw_residual_bounds: Dict[str, float] = {}
        hidden_gate_raw_residual_bounds: Dict[str, float] = {}
        input_delta_norm = 0.0
        hidden_delta_norm = 0.0
        for gate in GATE_NAMES:
            delta_x_gate = x_t - self.x_ref_gate[gate]
            delta_h_gate = self.h_prev - self.h_ref_gate[gate]
            input_delta_norm += float(torch.linalg.vector_norm(delta_x_gate).item())
            hidden_delta_norm += float(torch.linalg.vector_norm(delta_h_gate).item())
            input_raw = float(torch.sum(torch.abs(delta_x_gate) * self.input_gate_sensitivity[gate]).item())
            hidden_raw = float(torch.sum(torch.abs(delta_h_gate) * self.hidden_gate_sensitivity[gate]).item())
            input_gate_raw_residual_bounds[gate] = input_raw
            hidden_gate_raw_residual_bounds[gate] = hidden_raw
            input_gate_residual_bounds[gate] = self._normalize_gate_residual_bound(input_raw, "input", gate, delta_x_gate)
            hidden_gate_residual_bounds[gate] = self._normalize_gate_residual_bound(
                hidden_raw,
                "hidden",
                gate,
                delta_h_gate,
            )
        input_delta_norm /= float(len(GATE_NAMES))
        hidden_delta_norm /= float(len(GATE_NAMES))
        input_residual_bound = sum(input_gate_residual_bounds.values())
        hidden_residual_bound = sum(hidden_gate_residual_bounds.values())
        input_raw_residual_bound = sum(input_gate_raw_residual_bounds.values())
        hidden_raw_residual_bound = sum(hidden_gate_raw_residual_bounds.values())
        residual_bound = input_residual_bound + hidden_residual_bound
        refresh = bool(force_refresh)
        if residual_alpha is not None and self.step_index > 0:
            margin_budget = float(residual_alpha) * max(float(margin), 1e-3)
            refresh = refresh or residual_bound > margin_budget

        gate_input_active_ratios: Dict[str, float] = {}
        gate_hidden_active_ratios: Dict[str, float] = {}
        gate_input_active_indices: Dict[str, List[int]] = {}
        gate_hidden_active_indices: Dict[str, List[int]] = {}
        total_active_x = 0
        total_active_h = 0
        input_cols = int(self.weight_ih.shape[1])
        hidden_cols = int(self.weight_hh.shape[1])
        gate_rows = int(self.weight_ih.shape[0] // 3)

        for gate in GATE_NAMES:
            row_slice = self.gate_slices[gate]
            delta_x_gate = x_t - self.x_ref_gate[gate]
            delta_h_gate = self.h_prev - self.h_ref_gate[gate]
            active_x = torch.abs(delta_x_gate) > float(input_gate_thresholds[gate])
            active_h = torch.abs(delta_h_gate) > float(hidden_gate_thresholds[gate])
            if self.step_index == 0 or refresh:
                active_x = torch.ones_like(active_x, dtype=torch.bool)
                active_h = torch.ones_like(active_h, dtype=torch.bool)
            active_x_count = int(active_x.sum().item())
            active_h_count = int(active_h.sum().item())
            total_active_x += active_x_count
            total_active_h += active_h_count
            gate_input_active_ratios[gate] = active_x_count / max(1, input_cols)
            gate_hidden_active_ratios[gate] = active_h_count / max(1, hidden_cols)
            gate_input_active_indices[gate] = self._active_indices(active_x)
            gate_hidden_active_indices[gate] = self._active_indices(active_h)

            self.stats.add_delta_matvec(
                gate_rows,
                input_cols,
                active_x_count,
                input_side=True,
                activation_bits=spec_bits(qspec(self.variant, "input")),
                weight_bits=spec_bits(qspec(self.variant, "weight")),
            )
            self.stats.add_delta_matvec(
                gate_rows,
                hidden_cols,
                active_h_count,
                input_side=False,
                activation_bits=spec_bits(qspec(self.variant, "h")),
                weight_bits=spec_bits(qspec(self.variant, "weight")),
            )

            if active_x_count > 0:
                self.gi[row_slice] = self.gi[row_slice] + torch.mv(
                    self.weight_ih[row_slice, :][:, active_x],
                    delta_x_gate[active_x],
                )
                self.x_ref_gate[gate] = self.x_ref_gate[gate].clone()
                self.x_ref_gate[gate][active_x] = x_t[active_x]
            if active_h_count > 0:
                self.gh[row_slice] = self.gh[row_slice] + torch.mv(
                    self.weight_hh[row_slice, :][:, active_h],
                    delta_h_gate[active_h],
                )
                self.h_ref_gate[gate] = self.h_ref_gate[gate].clone()
                self.h_ref_gate[gate][active_h] = self.h_prev[active_h]

        traffic = self._traffic_bytes(total_active_x, total_active_h, gatewise=True)
        self.gi = fixed_quantize(self.gi, qspec(self.variant, "preact"))
        self.gh = fixed_quantize(self.gh, qspec(self.variant, "preact"))
        self.step_index += 1
        self.last_diag = StepDiagnostics(
            input_active_ratio=total_active_x / max(1, input_cols * len(GATE_NAMES)),
            hidden_active_ratio=total_active_h / max(1, hidden_cols * len(GATE_NAMES)),
            input_delta_norm=input_delta_norm,
            hidden_delta_norm=hidden_delta_norm,
            input_residual_bound=input_residual_bound,
            hidden_residual_bound=hidden_residual_bound,
            input_raw_residual_bound=input_raw_residual_bound,
            hidden_raw_residual_bound=hidden_raw_residual_bound,
            force_refresh=refresh,
            input_gate_active_ratios=gate_input_active_ratios,
            hidden_gate_active_ratios=gate_hidden_active_ratios,
            input_gate_residual_bounds=input_gate_residual_bounds,
            hidden_gate_residual_bounds=hidden_gate_residual_bounds,
            input_gate_raw_residual_bounds=input_gate_raw_residual_bounds,
            hidden_gate_raw_residual_bounds=hidden_gate_raw_residual_bounds,
            input_gate_active_indices=gate_input_active_indices,
            hidden_gate_active_indices=gate_hidden_active_indices,
            **traffic,
        )
        return self._apply_gru_update(self.gi, self.gh)

    def pending_residual_bound(self) -> float:
        if self.h_ref_gate is not None:
            total = 0.0
            for gate in GATE_NAMES:
                hidden_delta = self.h_prev - self.h_ref_gate[gate]
                raw = float(torch.sum(torch.abs(hidden_delta) * self.hidden_gate_sensitivity[gate]).item())
                total += self._normalize_gate_residual_bound(raw, "hidden", gate, hidden_delta)
            return float(total)
        if self.x_ref is None:
            input_bound = 0.0
        else:
            input_bound = float(torch.sum(torch.abs(self.x_ref) * 0.0).item())
        hidden_delta = self.h_prev - self.h_ref
        hidden_bound_raw = float(torch.sum(torch.abs(hidden_delta) * self.hidden_sensitivity).item())
        hidden_bound = self._normalize_residual_bound(hidden_bound_raw, "hidden", hidden_delta)
        return float(input_bound + hidden_bound)

    def _apply_gru_update(
        self,
        gi: torch.Tensor,
        gh: torch.Tensor,
        precision_bits: Optional[int] = None,
        precision_fake_quant: bool = False,
    ) -> torch.Tensor:
        i_r, i_z, i_n = gi.chunk(3, dim=0)
        h_r, h_z, h_n = gh.chunk(3, dim=0)
        r = apply_gate_activation(
            phase_fake_quantize(i_r + h_r, self.variant, "preact", precision_bits, precision_fake_quant),
            "sigmoid",
            "r",
            self.variant,
            self.lut,
            self.stats,
        )
        r = phase_fake_quantize(r, self.variant, "r", precision_bits, precision_fake_quant)
        z = apply_gate_activation(
            phase_fake_quantize(i_z + h_z, self.variant, "preact", precision_bits, precision_fake_quant),
            "sigmoid",
            "z",
            self.variant,
            self.lut,
            self.stats,
        )
        z = phase_fake_quantize(z, self.variant, "z", precision_bits, precision_fake_quant)
        n_pre = phase_fake_quantize(
            i_n + phase_fake_quantize(r * h_n, self.variant, "preact", precision_bits, precision_fake_quant),
            self.variant,
            "preact",
            precision_bits,
            precision_fake_quant,
        )
        n = apply_gate_activation(n_pre, "tanh", "n", self.variant, self.lut, self.stats)
        n = phase_fake_quantize(n, self.variant, "n", precision_bits, precision_fake_quant)
        h = phase_fake_quantize(
            phase_fake_quantize((1.0 - z) * n, self.variant, "h", precision_bits, precision_fake_quant)
            + phase_fake_quantize(z * self.h_prev, self.variant, "h", precision_bits, precision_fake_quant),
            self.variant,
            "h",
            precision_bits,
            precision_fake_quant,
        )
        self.h_prev = h
        return h

    def step(
        self,
        x_t: torch.Tensor,
        input_threshold: float,
        hidden_threshold: float,
        force_refresh: bool = False,
        residual_alpha: Optional[float] = None,
        margin: float = 0.0,
        input_gate_thresholds: Optional[Dict[str, float]] = None,
        hidden_gate_thresholds: Optional[Dict[str, float]] = None,
        budget_ratio: Optional[float] = None,
        precision_bits: Optional[int] = None,
        precision_fake_quant: bool = False,
    ) -> torch.Tensor:
        x_t = fixed_quantize(x_t, qspec(self.variant, "input"))
        if precision_fake_quant and precision_bits is not None:
            x_t = phase_fake_quantize(x_t, self.variant, "input", precision_bits, True)
            self.h_prev = phase_fake_quantize(self.h_prev, self.variant, "h", precision_bits, True)
        if input_gate_thresholds is not None and hidden_gate_thresholds is not None:
            return self._step_gatewise(
                x_t,
                input_gate_thresholds=input_gate_thresholds,
                hidden_gate_thresholds=hidden_gate_thresholds,
                force_refresh=force_refresh,
                residual_alpha=residual_alpha,
                margin=margin,
            )
        if self.x_ref is None:
            self.x_ref = torch.zeros_like(x_t)

        delta_x = x_t - self.x_ref
        delta_h = self.h_prev - self.h_ref
        input_raw_residual_bound = float(torch.sum(torch.abs(delta_x) * self.input_sensitivity).item())
        hidden_raw_residual_bound = float(torch.sum(torch.abs(delta_h) * self.hidden_sensitivity).item())
        input_residual_bound = self._normalize_residual_bound(input_raw_residual_bound, "input", delta_x)
        hidden_residual_bound = self._normalize_residual_bound(hidden_raw_residual_bound, "hidden", delta_h)
        input_gate_residual_bounds, input_gate_raw_residual_bounds = self._gate_residual_bounds(delta_x, "input")
        hidden_gate_residual_bounds, hidden_gate_raw_residual_bounds = self._gate_residual_bounds(delta_h, "hidden")
        residual_bound = input_residual_bound + hidden_residual_bound
        refresh = bool(force_refresh)
        if residual_alpha is not None and self.step_index > 0:
            margin_budget = float(residual_alpha) * max(float(margin), 1e-3)
            refresh = refresh or residual_bound > margin_budget

        active_x = torch.abs(delta_x) > float(input_threshold)
        active_h = torch.abs(delta_h) > float(hidden_threshold)
        if self.step_index == 0 or refresh:
            active_x = torch.ones_like(active_x, dtype=torch.bool)
            active_h = torch.ones_like(active_h, dtype=torch.bool)
        active_x = self._cap_active_by_budget(
            active_x,
            torch.abs(delta_x) * self.input_sensitivity,
            budget_ratio,
        )
        active_h = self._cap_active_by_budget(
            active_h,
            torch.abs(delta_h) * self.hidden_sensitivity,
            budget_ratio,
        )

        active_x_count = int(active_x.sum().item())
        active_h_count = int(active_h.sum().item())
        active_x_indices = self._active_indices(active_x)
        active_h_indices = self._active_indices(active_h)
        input_active_ratio = active_x_count / max(1, int(active_x.numel()))
        hidden_active_ratio = active_h_count / max(1, int(active_h.numel()))
        effective_bits = int(precision_bits) if precision_bits is not None else None
        dense_input_bits = spec_bits(qspec(self.variant, "input"))
        dense_hidden_bits = spec_bits(qspec(self.variant, "h"))
        dense_weight_bits = spec_bits(qspec(self.variant, "weight"))
        input_bits = effective_bits if effective_bits is not None else dense_input_bits
        hidden_bits = effective_bits if effective_bits is not None else dense_hidden_bits
        weight_bits = effective_bits if effective_bits is not None else dense_weight_bits
        input_bits = min(input_bits, dense_input_bits)
        hidden_bits = min(hidden_bits, dense_hidden_bits)
        weight_bits = min(weight_bits, dense_weight_bits)
        traffic = self._traffic_bytes(
            active_x_count,
            active_h_count,
            input_bits_override=input_bits,
            hidden_bits_override=hidden_bits,
            weight_bits_override=weight_bits,
        )
        add_delta_matvec_with_effective_bits(
            self.stats,
            self.weight_ih.shape[0],
            self.weight_ih.shape[1],
            active_x_count,
            input_side=True,
            dense_activation_bits=dense_input_bits,
            dense_weight_bits=dense_weight_bits,
            actual_activation_bits=input_bits,
            actual_weight_bits=weight_bits,
        )
        add_delta_matvec_with_effective_bits(
            self.stats,
            self.weight_hh.shape[0],
            self.weight_hh.shape[1],
            active_h_count,
            input_side=False,
            dense_activation_bits=dense_hidden_bits,
            dense_weight_bits=dense_weight_bits,
            actual_activation_bits=hidden_bits,
            actual_weight_bits=weight_bits,
        )

        if active_x_count > 0:
            weight_ih = self.weight_ih[:, active_x]
            delta_x_active = delta_x[active_x]
            if precision_fake_quant and precision_bits is not None:
                weight_ih = phase_fake_quantize(weight_ih, self.variant, "weight", precision_bits, True)
                delta_x_active = phase_fake_quantize(delta_x_active, self.variant, "input", precision_bits, True)
            self.gi = self.gi + torch.mv(weight_ih, delta_x_active)
            self.x_ref = self.x_ref.clone()
            self.x_ref[active_x] = x_t[active_x]
        if active_h_count > 0:
            weight_hh = self.weight_hh[:, active_h]
            delta_h_active = delta_h[active_h]
            if precision_fake_quant and precision_bits is not None:
                weight_hh = phase_fake_quantize(weight_hh, self.variant, "weight", precision_bits, True)
                delta_h_active = phase_fake_quantize(delta_h_active, self.variant, "h", precision_bits, True)
            self.gh = self.gh + torch.mv(weight_hh, delta_h_active)
            self.h_ref = self.h_ref.clone()
            self.h_ref[active_h] = self.h_prev[active_h]

        self.gi = phase_fake_quantize(self.gi, self.variant, "preact", precision_bits, precision_fake_quant)
        self.gh = phase_fake_quantize(self.gh, self.variant, "preact", precision_bits, precision_fake_quant)
        self.step_index += 1
        self.last_diag = StepDiagnostics(
            input_active_ratio=input_active_ratio,
            hidden_active_ratio=hidden_active_ratio,
            input_delta_norm=float(torch.linalg.vector_norm(delta_x).item()),
            hidden_delta_norm=float(torch.linalg.vector_norm(delta_h).item()),
            input_residual_bound=input_residual_bound,
            hidden_residual_bound=hidden_residual_bound,
            input_raw_residual_bound=input_raw_residual_bound,
            hidden_raw_residual_bound=hidden_raw_residual_bound,
            force_refresh=refresh,
            input_gate_active_ratios={gate: input_active_ratio for gate in GATE_NAMES},
            hidden_gate_active_ratios={gate: hidden_active_ratio for gate in GATE_NAMES},
            input_gate_residual_bounds=input_gate_residual_bounds,
            hidden_gate_residual_bounds=hidden_gate_residual_bounds,
            input_gate_raw_residual_bounds=input_gate_raw_residual_bounds,
            hidden_gate_raw_residual_bounds=hidden_gate_raw_residual_bounds,
            input_gate_active_indices={gate: list(active_x_indices) for gate in GATE_NAMES},
            hidden_gate_active_indices={gate: list(active_h_indices) for gate in GATE_NAMES},
            **traffic,
        )
        return self._apply_gru_update(
            self.gi,
            self.gh,
            precision_bits=precision_bits,
            precision_fake_quant=precision_fake_quant,
        )


def normalized_edit_alignment(reference: str, hypothesis: str) -> Tuple[List[int], List[int]]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = i
        back[i][0] = "D"
    for j in range(1, cols):
        cost[0][j] = j
        back[0][j] = "I"
    for i in range(1, rows):
        for j in range(1, cols):
            sub = cost[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1)
            delete = cost[i - 1][j] + 1
            insert = cost[i][j - 1] + 1
            best = min(sub, delete, insert)
            cost[i][j] = best
            back[i][j] = "M" if best == sub else ("D" if best == delete else "I")
    ref_error = [1] * len(reference)
    hyp_error = [1] * len(hypothesis)
    i = len(reference)
    j = len(hypothesis)
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            is_error = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            ref_error[i - 1] = is_error
            hyp_error[j - 1] = is_error
            i -= 1
            j -= 1
        elif op == "D":
            ref_error[i - 1] = 1
            i -= 1
        else:
            hyp_error[j - 1] = 1
            j -= 1
    return ref_error, hyp_error


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


def predicted_events(
    logits: torch.Tensor,
    length: int,
    start_threshold: float,
    start_delay: int,
) -> List[Tuple[int, str, int]]:
    from paper_v5_pipeline import CHAR_DEF

    usable = logits[:length]
    if usable.numel() == 0:
        return []
    start_prob = torch.sigmoid(usable[:, -1])
    best_class = torch.argmax(usable[:, :-1], dim=-1)
    transitions = torch.argwhere((start_prob[:-1] < start_threshold) & (start_prob[1:] > start_threshold)).reshape(-1)
    events: List[Tuple[int, str, int]] = []
    for event_idx, idx in enumerate(transitions.tolist()):
        char_t = min(int(idx) + int(start_delay), int(usable.shape[0]) - 1)
        char = CHAR_DEF["charListAbbr"][int(best_class[char_t].item())]
        events.append((char_t, char, event_idx))
    return events


def add_dense_frontend_stats(
    stats: HardwareStats,
    model: PaperBrainToTextV5,
    length: int,
    variant: VariantConfig,
) -> None:
    stats.add_dense_matvec_batch(
        length,
        int(model.session_weight.shape[1]),
        int(model.session_weight.shape[2]),
        activation_bits=spec_bits(qspec(variant, "input")),
        weight_bits=spec_bits(variant.fixed_point),
    )
    stats.add_dense_matvec_batch(
        length,
        int(model.smoother.channels),
        int(model.smoother.kernel.shape[-1]),
        activation_bits=spec_bits(qspec(variant, "input")),
        weight_bits=spec_bits(variant.fixed_point),
    )


def semantic_forward_one(
    model: PaperBrainToTextV5,
    inputs: torch.Tensor,
    length: int,
    session_id: int,
    variant: VariantConfig,
    scheduler: SchedulerConfig,
    device: torch.device,
    trace_context: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, HardwareStats, List[Dict[str, object]]]:
    stats = HardwareStats()
    lut = LutActivations(size=variant.lut_size, value_range=variant.lut_range, device=device)
    trace_rows: List[Dict[str, object]] = []
    x = inputs[:length].to(device)
    session_tensor = torch.tensor([session_id], dtype=torch.long, device=device)

    with torch.no_grad():
        weights = model.session_weight[session_tensor][0]
        bias = model.session_bias[session_tensor][0]
        add_dense_frontend_stats(stats, model, length, variant)
        x = x @ weights.T + bias
        x = F.conv1d(
            x.unsqueeze(0).transpose(1, 2),
            model.smoother.kernel.to(device),
            padding=model.smoother.padding,
            groups=model.smoother.channels,
        ).transpose(1, 2).squeeze(0)
        x = fixed_quantize(x, qspec(variant, "input"))

        h0_1 = model.start_state_1[0, 0].to(device)
        h0_2 = model.start_state_2[0, 0].to(device)
        gru1 = InstrumentedGruRuntime(
            model.gru1.weight_ih_l0.to(device),
            model.gru1.weight_hh_l0.to(device),
            model.gru1.bias_ih_l0.to(device),
            model.gru1.bias_hh_l0.to(device),
            h0_1,
            variant,
            lut,
            stats,
            drift_normalizer=scheduler.drift_normalizer,
        )
        gru2 = InstrumentedGruRuntime(
            model.gru2.weight_ih_l0.to(device),
            model.gru2.weight_hh_l0.to(device),
            model.gru2.bias_ih_l0.to(device),
            model.gru2.bias_hh_l0.to(device),
            h0_2,
            variant,
            lut,
            stats,
            drift_normalizer=scheduler.drift_normalizer,
        )
        readout_weight = fixed_quantize(model.readout.weight.to(device), qspec(variant, "readout"))
        readout_bias = fixed_quantize(model.readout.bias.to(device), qspec(variant, "readout"))

        previous_start_prob = 0.0
        previous_margin = 999.0
        logits_down: List[torch.Tensor] = []
        pending_logits: Optional[torch.Tensor] = None
        mode_counts: Dict[str, int] = {}
        refresh_count = 0
        trace_context = trace_context or {}

        for t in range(length):
            residual_bound = gru1.pending_residual_bound() + gru2.pending_residual_bound()
            schedule_start_prob = previous_start_prob
            schedule_margin = previous_margin
            mode = choose_mode(previous_start_prob, previous_margin, residual_bound, scheduler)
            input_threshold, hidden_threshold = mode_thresholds(mode, scheduler)
            input_gate_thresholds, hidden_gate_thresholds = gate_mode_thresholds(mode, scheduler)
            force_refresh = mode == "REFRESH"
            alpha = mode_alpha(mode, scheduler) if scheduler.enable_drift_bound else None
            semantic_phase = semantic_phase_from_signals(schedule_start_prob, schedule_margin, mode, scheduler)
            budget_ratio = mode_budget_ratio(mode, scheduler)
            precision_bits = mode_precision_bits(mode, scheduler)
            precision_fake_quant = bool(scheduler.precision_fake_quant)
            h1 = gru1.step(
                x[t],
                input_threshold=input_threshold,
                hidden_threshold=hidden_threshold,
                force_refresh=force_refresh,
                residual_alpha=alpha,
                margin=previous_margin,
                input_gate_thresholds=input_gate_thresholds,
                hidden_gate_thresholds=hidden_gate_thresholds,
                budget_ratio=budget_ratio,
                precision_bits=precision_bits,
                precision_fake_quant=precision_fake_quant,
            )
            h2_diag = StepDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
            did_gru2 = (t % int(model.skip_len)) == 0
            current_start_prob = previous_start_prob
            current_margin = previous_margin
            if did_gru2:
                h2 = gru2.step(
                    h1,
                    input_threshold=input_threshold,
                    hidden_threshold=hidden_threshold,
                    force_refresh=force_refresh,
                    residual_alpha=alpha,
                    margin=previous_margin,
                    input_gate_thresholds=input_gate_thresholds,
                    hidden_gate_thresholds=hidden_gate_thresholds,
                    budget_ratio=budget_ratio,
                    precision_bits=precision_bits,
                    precision_fake_quant=precision_fake_quant,
                )
                h2_diag = gru2.last_diag
                dense_readout_activation_bits = spec_bits(qspec(variant, "h"))
                dense_readout_weight_bits = spec_bits(qspec(variant, "readout"))
                readout_activation_bits = (
                    int(precision_bits) if precision_bits is not None else dense_readout_activation_bits
                )
                readout_weight_bits = int(precision_bits) if precision_bits is not None else dense_readout_weight_bits
                readout_activation_bits = min(readout_activation_bits, dense_readout_activation_bits)
                readout_weight_bits = min(readout_weight_bits, dense_readout_weight_bits)
                add_dense_matvec_with_effective_bits(
                    stats,
                    readout_weight.shape[0],
                    readout_weight.shape[1],
                    dense_activation_bits=dense_readout_activation_bits,
                    dense_weight_bits=dense_readout_weight_bits,
                    actual_activation_bits=readout_activation_bits,
                    actual_weight_bits=readout_weight_bits,
                )
                readout_h = phase_fake_quantize(h2, variant, "h", precision_bits, precision_fake_quant)
                readout_w = (
                    phase_fake_quantize(readout_weight, variant, "readout", precision_bits, True)
                    if precision_fake_quant and precision_bits is not None
                    else readout_weight
                )
                logits_t = F.linear(readout_h, readout_w, readout_bias)
                logits_t = phase_fake_quantize(logits_t, variant, "readout", precision_bits, precision_fake_quant)
                logits_down.append(logits_t)
                pending_logits = logits_t
                current_start_prob = float(torch.sigmoid(logits_t[-1]).item())
                current_margin = top_margin(logits_t)
                previous_start_prob = current_start_prob
                previous_margin = current_margin

            diag = gru1.last_diag
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            refresh_count += int(diag.force_refresh or h2_diag.force_refresh)
            if trace_context.get("emit_active_sets", False) and variant_matches_filter(
                str(trace_context.get("variant", "")),
                str(trace_context.get("active_set_variant_filter", "")),
            ):
                record_active_sets_for_layer(
                    trace_context,
                    t=t,
                    downsample_step=int(t // int(model.skip_len)),
                    layer="gru1",
                    diag=diag,
                    mode=mode,
                    semantic_phase=semantic_phase,
                    input_total=int(gru1.weight_ih.shape[1]),
                    hidden_total=int(gru1.weight_hh.shape[1]),
                    input_threshold=input_threshold,
                    hidden_threshold=hidden_threshold,
                    input_gate_thresholds=input_gate_thresholds,
                    hidden_gate_thresholds=hidden_gate_thresholds,
                )
                if did_gru2:
                    record_active_sets_for_layer(
                        trace_context,
                        t=t,
                        downsample_step=int(t // int(model.skip_len)),
                        layer="gru2",
                        diag=h2_diag,
                        mode=mode,
                        semantic_phase=semantic_phase,
                        input_total=int(gru2.weight_ih.shape[1]),
                        hidden_total=int(gru2.weight_hh.shape[1]),
                        input_threshold=input_threshold,
                        hidden_threshold=hidden_threshold,
                        input_gate_thresholds=input_gate_thresholds,
                        hidden_gate_thresholds=hidden_gate_thresholds,
                    )
            if trace_context.get("record_trace", False):
                trace_rows.append(
                    {
                        **{
                            k: v
                            for k, v in trace_context.items()
                            if k
                            not in {
                                "record_trace",
                                "emit_active_sets",
                                "active_set_records",
                                "max_active_set_records",
                                "active_set_variant_filter",
                            }
                        },
                        "t": t,
                        "downsample_step": int(t // int(model.skip_len)),
                        "mode": mode,
                        "semantic_phase": semantic_phase,
                        "input_threshold": input_threshold,
                        "hidden_threshold": hidden_threshold,
                        "gate_policy": scheduler.gate_policy,
                        "active_budget_policy": scheduler.active_budget_policy,
                        "budget_ratio": budget_ratio if budget_ratio is not None else "",
                        "precision_policy": scheduler.precision_policy,
                        "precision_fake_quant": int(precision_fake_quant),
                        "precision_bits": precision_bits if precision_bits is not None else "",
                        "schedule_start_prob": schedule_start_prob,
                        "schedule_margin": schedule_margin,
                        "start_prob": current_start_prob,
                        "top1_top2_margin": current_margin,
                        "target_char_start": "",
                        "predicted_start": "",
                        "predicted_error_event": "",
                        "near_target_start": "",
                        "near_predicted_start": "",
                        "gru1_input_delta_norm": diag.input_delta_norm,
                        "gru1_hidden_delta_norm": diag.hidden_delta_norm,
                        "gru1_input_active_ratio": diag.input_active_ratio,
                        "gru1_hidden_active_ratio": diag.hidden_active_ratio,
                        "gru1_input_residual_bound": diag.input_residual_bound,
                        "gru1_hidden_residual_bound": diag.hidden_residual_bound,
                        "gru1_input_raw_residual_bound": diag.input_raw_residual_bound,
                        "gru1_hidden_raw_residual_bound": diag.hidden_raw_residual_bound,
                        "gru2_ran": int(did_gru2),
                        "gru2_input_delta_norm": h2_diag.input_delta_norm if did_gru2 else "",
                        "gru2_hidden_delta_norm": h2_diag.hidden_delta_norm if did_gru2 else "",
                        "gru2_input_active_ratio": h2_diag.input_active_ratio if did_gru2 else "",
                        "gru2_hidden_active_ratio": h2_diag.hidden_active_ratio if did_gru2 else "",
                        "gru2_input_residual_bound": h2_diag.input_residual_bound if did_gru2 else "",
                        "gru2_hidden_residual_bound": h2_diag.hidden_residual_bound if did_gru2 else "",
                        "gru2_input_raw_residual_bound": h2_diag.input_raw_residual_bound if did_gru2 else "",
                        "gru2_hidden_raw_residual_bound": h2_diag.hidden_raw_residual_bound if did_gru2 else "",
                        "total_residual_bound": residual_bound,
                        "current_total_residual_bound": diag.input_residual_bound
                        + diag.hidden_residual_bound
                        + (h2_diag.input_residual_bound if did_gru2 else 0.0)
                        + (h2_diag.hidden_residual_bound if did_gru2 else 0.0),
                        "current_total_raw_residual_bound": diag.input_raw_residual_bound
                        + diag.hidden_raw_residual_bound
                        + (h2_diag.input_raw_residual_bound if did_gru2 else 0.0)
                        + (h2_diag.hidden_raw_residual_bound if did_gru2 else 0.0),
                        "drift_normalizer": scheduler.drift_normalizer,
                        **diagnostic_trace_fields("gru1", diag),
                        **diagnostic_trace_fields("gru2", h2_diag, include=did_gru2),
                        "total_actual_weight_bytes": diag.actual_weight_bytes
                        + (h2_diag.actual_weight_bytes if did_gru2 else 0.0),
                        "total_dense_weight_bytes": diag.dense_weight_bytes
                        + (h2_diag.dense_weight_bytes if did_gru2 else 0.0),
                        "total_activation_bytes": diag.input_activation_bytes
                        + diag.hidden_activation_bytes
                        + (
                            h2_diag.input_activation_bytes + h2_diag.hidden_activation_bytes
                            if did_gru2
                            else 0.0
                        ),
                        "total_reference_write_bytes": diag.reference_write_bytes
                        + (h2_diag.reference_write_bytes if did_gru2 else 0.0),
                        "total_hidden_state_write_bytes": diag.hidden_state_write_bytes
                        + (h2_diag.hidden_state_write_bytes if did_gru2 else 0.0),
                        "total_metadata_bytes": diag.metadata_bytes + (h2_diag.metadata_bytes if did_gru2 else 0.0),
                        "total_traffic_bytes": diag.total_traffic_bytes
                        + (h2_diag.total_traffic_bytes if did_gru2 else 0.0),
                    }
                )

        if not logits_down:
            logits_down.append(torch.zeros((readout_weight.shape[0],), device=device))
        logits_down_tensor = torch.stack(logits_down, dim=0)
        logits = torch.repeat_interleave(logits_down_tensor, repeats=model.skip_len, dim=0)[:length]
        stats.add_blank_active_ratio(mode_counts.get("PROTECT", 0) / max(1, length))
        stats.notes.append(
            json.dumps(
                {
                    "mode_counts": mode_counts,
                    "forced_refresh_steps": refresh_count,
                },
                sort_keys=True,
            )
        )
    return logits.detach().cpu(), stats, trace_rows


def annotate_trace_events(
    trace_rows: List[Dict[str, object]],
    reference: str,
    logits: torch.Tensor,
    char_start_target: torch.Tensor,
    start_threshold: float,
    start_delay: int,
    radius: int,
) -> None:
    if not trace_rows:
        return
    length = len(trace_rows)
    target_starts = set(torch.argwhere(char_start_target[:length] > 0.5).reshape(-1).tolist())
    events = predicted_events(logits, length, start_threshold, start_delay)
    hypothesis = "".join(char for _, char, _ in events)
    _, hyp_errors = normalized_edit_alignment(reference, hypothesis)
    pred_by_t: Dict[int, Tuple[str, int]] = {}
    for char_t, char, event_idx in events:
        pred_by_t[int(char_t)] = (char, hyp_errors[event_idx] if event_idx < len(hyp_errors) else 1)
    target_near = set()
    for start in target_starts:
        for t in range(max(0, int(start) - radius), min(length, int(start) + radius + 1)):
            target_near.add(t)
    pred_near = set()
    for char_t, _char, _idx in events:
        for t in range(max(0, int(char_t) - radius), min(length, int(char_t) + radius + 1)):
            pred_near.add(t)
    for row in trace_rows:
        t = int(row["t"])
        row["target_char_start"] = int(t in target_starts)
        row["predicted_start"] = int(t in pred_by_t)
        row["predicted_error_event"] = int(pred_by_t.get(t, ("", 0))[1]) if t in pred_by_t else 0
        row["near_target_start"] = int(t in target_near)
        row["near_predicted_start"] = int(t in pred_near)


def aggregate_trace(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, Dict[str, float]] = {}
    for row in trace_rows:
        keys = [
            f"mode={row.get('mode')}",
            f"near_target={row.get('near_target_start')}",
            f"near_pred={row.get('near_predicted_start')}",
            f"pred_error={row.get('predicted_error_event')}",
        ]
        for key in keys:
            bucket = groups.setdefault(
                key,
                {
                    "group": key,
                    "steps": 0.0,
                    "mean_start_prob": 0.0,
                    "mean_margin": 0.0,
                    "mean_gru1_input_active": 0.0,
                    "mean_gru1_hidden_active": 0.0,
                    "mean_gru1_residual_bound": 0.0,
                    "mean_gru2_input_active": 0.0,
                    "mean_gru2_hidden_active": 0.0,
                    "gru2_steps": 0.0,
                    "mean_total_traffic_bytes": 0.0,
                    "mean_total_actual_weight_bytes": 0.0,
                    "mean_total_metadata_bytes": 0.0,
                    "mean_total_activation_bytes": 0.0,
                    "mean_total_reference_write_bytes": 0.0,
                    "mean_total_hidden_state_write_bytes": 0.0,
                    "mean_gru1_input_r_raw_residual": 0.0,
                    "mean_gru1_input_z_raw_residual": 0.0,
                    "mean_gru1_input_n_raw_residual": 0.0,
                    "mean_gru1_hidden_r_raw_residual": 0.0,
                    "mean_gru1_hidden_z_raw_residual": 0.0,
                    "mean_gru1_hidden_n_raw_residual": 0.0,
                },
            )
            bucket["steps"] += 1.0
            bucket["mean_start_prob"] += float(row.get("start_prob") or 0.0)
            bucket["mean_margin"] += float(row.get("top1_top2_margin") or 0.0)
            bucket["mean_gru1_input_active"] += float(row.get("gru1_input_active_ratio") or 0.0)
            bucket["mean_gru1_hidden_active"] += float(row.get("gru1_hidden_active_ratio") or 0.0)
            bucket["mean_gru1_residual_bound"] += float(row.get("gru1_input_residual_bound") or 0.0) + float(
                row.get("gru1_hidden_residual_bound") or 0.0
            )
            bucket["mean_total_traffic_bytes"] += float(row.get("total_traffic_bytes") or 0.0)
            bucket["mean_total_actual_weight_bytes"] += float(row.get("total_actual_weight_bytes") or 0.0)
            bucket["mean_total_metadata_bytes"] += float(row.get("total_metadata_bytes") or 0.0)
            bucket["mean_total_activation_bytes"] += float(row.get("total_activation_bytes") or 0.0)
            bucket["mean_total_reference_write_bytes"] += float(row.get("total_reference_write_bytes") or 0.0)
            bucket["mean_total_hidden_state_write_bytes"] += float(row.get("total_hidden_state_write_bytes") or 0.0)
            for gate in GATE_NAMES:
                bucket[f"mean_gru1_input_{gate}_raw_residual"] += float(
                    row.get(f"gru1_input_{gate}_raw_residual_bound") or 0.0
                )
                bucket[f"mean_gru1_hidden_{gate}_raw_residual"] += float(
                    row.get(f"gru1_hidden_{gate}_raw_residual_bound") or 0.0
                )
            if row.get("gru2_ran"):
                bucket["gru2_steps"] += 1.0
                bucket["mean_gru2_input_active"] += float(row.get("gru2_input_active_ratio") or 0.0)
                bucket["mean_gru2_hidden_active"] += float(row.get("gru2_hidden_active_ratio") or 0.0)
    out = []
    for bucket in groups.values():
        steps = max(1.0, bucket["steps"])
        gru2_steps = max(1.0, bucket["gru2_steps"])
        row = dict(bucket)
        for key in [
            "mean_start_prob",
            "mean_margin",
            "mean_gru1_input_active",
            "mean_gru1_hidden_active",
            "mean_gru1_residual_bound",
            "mean_total_traffic_bytes",
            "mean_total_actual_weight_bytes",
            "mean_total_metadata_bytes",
            "mean_total_activation_bytes",
            "mean_total_reference_write_bytes",
            "mean_total_hidden_state_write_bytes",
            "mean_gru1_input_r_raw_residual",
            "mean_gru1_input_z_raw_residual",
            "mean_gru1_input_n_raw_residual",
            "mean_gru1_hidden_r_raw_residual",
            "mean_gru1_hidden_z_raw_residual",
            "mean_gru1_hidden_n_raw_residual",
        ]:
            row[key] = row[key] / steps
        row["mean_gru2_input_active"] = row["mean_gru2_input_active"] / gru2_steps
        row["mean_gru2_hidden_active"] = row["mean_gru2_hidden_active"] / gru2_steps
        out.append(row)
    return sorted(out, key=lambda item: str(item["group"]))


def append_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def semantic_phase_from_signals(
    start_prob: float,
    margin: float,
    mode: str,
    scheduler: SchedulerConfig,
) -> str:
    if float(start_prob) > float(scheduler.high_start_threshold):
        return "char_start"
    if float(margin) < float(scheduler.low_margin_threshold):
        return "low_margin"
    if float(start_prob) < float(scheduler.blank_threshold) and float(margin) > float(scheduler.high_margin_threshold):
        return "blank_stable"
    if float(start_prob) >= float(scheduler.blank_threshold) and float(margin) > float(scheduler.high_margin_threshold):
        return "high_margin_nonblank"
    if mode == "AGGRESSIVE":
        return "blank_stable"
    return "normal"


def parse_str_list(raw: Optional[str], fallback: str) -> List[str]:
    if raw is None or str(raw).strip() == "":
        return [str(fallback)]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def variant_matches_filter(variant_name: str, raw_filter: str) -> bool:
    tokens = [item.strip() for item in str(raw_filter or "").split(",") if item.strip()]
    if not tokens:
        return True
    return any(token in str(variant_name) for token in tokens)


def record_active_sets_for_layer(
    trace_context: Dict[str, object],
    *,
    t: int,
    downsample_step: int,
    layer: str,
    diag: StepDiagnostics,
    mode: str,
    semantic_phase: str,
    input_total: int,
    hidden_total: int,
    input_threshold: float,
    hidden_threshold: float,
    input_gate_thresholds: Optional[Dict[str, float]],
    hidden_gate_thresholds: Optional[Dict[str, float]],
) -> None:
    records = trace_context.get("active_set_records")
    if not isinstance(records, list):
        return
    max_records = int(trace_context.get("max_active_set_records", 0) or 0)
    if max_records > 0 and len(records) >= max_records:
        return
    common = {
        key: value
        for key, value in trace_context.items()
        if key
        not in {
            "record_trace",
            "emit_active_sets",
            "active_set_records",
            "max_active_set_records",
            "active_set_variant_filter",
        }
    }
    for gate in GATE_NAMES:
        for path, total, active, residual, raw_residual, threshold in [
            (
                "input",
                input_total,
                diag.input_gate_active_indices.get(gate, []),
                diag.input_gate_residual_bounds.get(gate, 0.0),
                diag.input_gate_raw_residual_bounds.get(gate, 0.0),
                (input_gate_thresholds or {}).get(gate, input_threshold),
            ),
            (
                "hidden",
                hidden_total,
                diag.hidden_gate_active_indices.get(gate, []),
                diag.hidden_gate_residual_bounds.get(gate, 0.0),
                diag.hidden_gate_raw_residual_bounds.get(gate, 0.0),
                (hidden_gate_thresholds or {}).get(gate, hidden_threshold),
            ),
        ]:
            records.append(
                {
                    **common,
                    "t": int(t),
                    "downsample_step": int(downsample_step),
                    "layer": layer,
                    "gate": gate,
                    "path": path,
                    "semantic_phase": semantic_phase,
                    "fsm_mode": mode,
                    "active_count": int(len(active)),
                    "total_count": int(total),
                    "active_ratio": float(len(active)) / max(1, int(total)),
                    "threshold": float(threshold),
                    "residual_bound": float(residual),
                    "raw_residual_bound": float(raw_residual),
                    "active_indices": list(active),
                }
            )
            if max_records > 0 and len(records) >= max_records:
                return


def save_active_set_outputs(output_dir: Path, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    metadata_rows: List[Dict[str, object]] = []
    packed_chunks: List[np.ndarray] = []
    packed_offsets = [0]
    total_counts: List[int] = []
    active_counts: List[int] = []
    indices_flat: List[int] = []
    indices_offsets = [0]
    for row_id, record in enumerate(records):
        active_indices = [int(idx) for idx in record.get("active_indices", [])]
        total_count = int(record.get("total_count", 0) or 0)
        mask = np.zeros(max(0, total_count), dtype=np.uint8)
        if active_indices and total_count > 0:
            valid = [idx for idx in active_indices if 0 <= idx < total_count]
            mask[np.asarray(valid, dtype=np.int64)] = 1
        packed = np.packbits(mask, bitorder="little")
        packed_chunks.append(packed)
        packed_offsets.append(packed_offsets[-1] + int(packed.size))
        total_counts.append(total_count)
        active_counts.append(len(active_indices))
        indices_flat.extend(active_indices)
        indices_offsets.append(indices_offsets[-1] + len(active_indices))
        meta = {key: value for key, value in record.items() if key != "active_indices"}
        meta["row_id"] = row_id
        metadata_rows.append(meta)

    write_csv(output_dir / "gate_active_trace.csv", metadata_rows)
    packed_masks = np.concatenate(packed_chunks).astype(np.uint8) if packed_chunks else np.asarray([], dtype=np.uint8)
    np.savez_compressed(
        output_dir / "active_set_trace.npz",
        packed_masks=packed_masks,
        packed_offsets=np.asarray(packed_offsets, dtype=np.int64),
        total_counts=np.asarray(total_counts, dtype=np.int32),
        active_counts=np.asarray(active_counts, dtype=np.int32),
        indices_flat=np.asarray(indices_flat, dtype=np.int32),
        indices_offsets=np.asarray(indices_offsets, dtype=np.int64),
        row_ids=np.arange(len(metadata_rows), dtype=np.int64),
    )


def make_variant() -> VariantConfig:
    kwargs = mixed_precision_kwargs("balanced")
    return VariantConfig(
        name="semantic_delta_mixed_balanced_lut",
        activation_mode="lut",
        lut_size=1024,
        lut_range=8.0,
        delta_input_threshold=0.08,
        delta_hidden_threshold=0.02,
        use_delta_gru=True,
        **kwargs,
    )


def parse_float_list(raw: Optional[str], fallback: float) -> List[float]:
    if raw is None or str(raw).strip() == "":
        return [float(fallback)]
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def requested_variants(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def fmt_float(value: float) -> str:
    text = f"{float(value):.4g}"
    return text.replace("-", "m").replace(".", "p")


def scheduler_name(base_name: str, config: SchedulerConfig, include_params: bool) -> str:
    if not include_params:
        return base_name
    parts = [
        f"hs{fmt_float(config.high_start_threshold)}",
        f"bt{fmt_float(config.blank_threshold)}",
        f"lm{fmt_float(config.low_margin_threshold)}",
        f"hm{fmt_float(config.high_margin_threshold)}",
        f"ps{fmt_float(config.protect_scale)}",
        f"as{fmt_float(config.aggressive_scale)}",
    ]
    if config.gate_policy != "uniform":
        parts.append(f"gp{config.gate_policy}")
    if config.signal_policy != "semantic":
        parts.append(f"sp{config.signal_policy}")
    if config.active_budget_policy != "none":
        parts.extend(
            [
                f"bp{config.active_budget_policy}",
                f"ka{fmt_float(config.budget_aggressive_ratio)}",
                f"kn{fmt_float(config.budget_normal_ratio)}",
                f"kp{fmt_float(config.budget_protect_ratio)}",
            ]
        )
    if config.precision_policy != "none":
        parts.extend(
            [
                f"pp{config.precision_policy}",
                f"fq{int(bool(config.precision_fake_quant))}",
                f"ba{int(config.precision_aggressive_bits)}",
                f"bn{int(config.precision_normal_bits)}",
                f"bp{int(config.precision_protect_bits)}",
            ]
        )
    if config.enable_drift_bound:
        parts.extend(
            [
                f"db{fmt_float(config.drift_budget)}",
                f"pa{fmt_float(config.protect_alpha)}",
                f"na{fmt_float(config.normal_alpha)}",
                f"aa{fmt_float(config.aggressive_alpha)}",
                f"dn{config.drift_normalizer}",
            ]
        )
    return f"{base_name}__{'_'.join(parts)}"


def optional_float(row: Dict[str, str], key: str, fallback: float) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return float(fallback)
    return float(value)


def scheduler_rows_from_csv(path: str, args: argparse.Namespace) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            rows.append(
                {
                    "name": row.get("name", f"cfg{idx}"),
                    "high_start_threshold": optional_float(row, "high_start_threshold", args.high_start_threshold),
                    "blank_threshold": optional_float(row, "blank_threshold", args.blank_threshold),
                    "low_margin_threshold": optional_float(row, "low_margin_threshold", args.low_margin_threshold),
                    "high_margin_threshold": optional_float(row, "high_margin_threshold", args.high_margin_threshold),
                    "protect_scale": optional_float(row, "protect_scale", args.protect_scale),
                    "aggressive_scale": optional_float(row, "aggressive_scale", args.aggressive_scale),
                    "drift_budget": optional_float(row, "drift_budget", args.drift_budget),
                    "protect_alpha": optional_float(row, "protect_alpha", args.protect_alpha),
                    "normal_alpha": optional_float(row, "normal_alpha", args.normal_alpha),
                    "aggressive_alpha": optional_float(row, "aggressive_alpha", args.aggressive_alpha),
                    "gate_policy": row.get("gate_policy", args.gate_policy),
                    "budget_aggressive_ratio": optional_float(
                        row, "budget_aggressive_ratio", args.budget_aggressive_ratio
                    ),
                    "budget_normal_ratio": optional_float(row, "budget_normal_ratio", args.budget_normal_ratio),
                    "budget_protect_ratio": optional_float(row, "budget_protect_ratio", args.budget_protect_ratio),
                    "precision_fake_quant": int(optional_float(row, "precision_fake_quant", int(bool(args.precision_fake_quant)))),
                    "precision_aggressive_bits": int(
                        optional_float(row, "precision_aggressive_bits", args.precision_aggressive_bits)
                    ),
                    "precision_normal_bits": int(optional_float(row, "precision_normal_bits", args.precision_normal_bits)),
                    "precision_protect_bits": int(
                        optional_float(row, "precision_protect_bits", args.precision_protect_bits)
                    ),
                }
            )
    return rows


def scheduler_grid(args: argparse.Namespace) -> List[Tuple[str, SchedulerConfig]]:
    requested = requested_variants(args.variants)
    requested_set = set(requested)
    scalar_base = {
        "base_input_threshold": args.base_input_threshold,
        "base_hidden_threshold": args.base_hidden_threshold,
        "protect_scale": args.protect_scale,
        "aggressive_scale": args.aggressive_scale,
        "high_start_threshold": args.high_start_threshold,
        "blank_threshold": args.blank_threshold,
        "low_margin_threshold": args.low_margin_threshold,
        "high_margin_threshold": args.high_margin_threshold,
        "drift_budget": args.drift_budget,
        "protect_alpha": args.protect_alpha,
        "normal_alpha": args.normal_alpha,
        "aggressive_alpha": args.aggressive_alpha,
        "drift_normalizer": args.drift_normalizer,
        "budget_aggressive_ratio": args.budget_aggressive_ratio,
        "budget_normal_ratio": args.budget_normal_ratio,
        "budget_protect_ratio": args.budget_protect_ratio,
        "precision_aggressive_bits": int(args.precision_aggressive_bits),
        "precision_normal_bits": int(args.precision_normal_bits),
        "precision_protect_bits": int(args.precision_protect_bits),
    }
    if args.scheduler_config_csv:
        sweep_rows = scheduler_rows_from_csv(args.scheduler_config_csv, args)
    else:
        sweep_rows = [
            {
                "name": "",
                "high_start_threshold": high_start_threshold,
                "blank_threshold": blank_threshold,
                "low_margin_threshold": low_margin_threshold,
                "high_margin_threshold": high_margin_threshold,
                "protect_scale": protect_scale,
                "aggressive_scale": aggressive_scale,
                "drift_budget": drift_budget,
                "protect_alpha": protect_alpha,
                "normal_alpha": normal_alpha,
                "aggressive_alpha": aggressive_alpha,
                "budget_aggressive_ratio": args.budget_aggressive_ratio,
                "budget_normal_ratio": args.budget_normal_ratio,
                "budget_protect_ratio": args.budget_protect_ratio,
                "precision_fake_quant": int(bool(args.precision_fake_quant)),
                "precision_aggressive_bits": int(args.precision_aggressive_bits),
                "precision_normal_bits": int(args.precision_normal_bits),
                "precision_protect_bits": int(args.precision_protect_bits),
            }
            for (
                high_start_threshold,
                blank_threshold,
                low_margin_threshold,
                high_margin_threshold,
                protect_scale,
                aggressive_scale,
                drift_budget,
                protect_alpha,
                normal_alpha,
                aggressive_alpha,
            ) in itertools.product(
                parse_float_list(args.high_start_thresholds, args.high_start_threshold),
                parse_float_list(args.blank_thresholds, args.blank_threshold),
                parse_float_list(args.low_margin_thresholds, args.low_margin_threshold),
                parse_float_list(args.high_margin_thresholds, args.high_margin_threshold),
                parse_float_list(args.protect_scales, args.protect_scale),
                parse_float_list(args.aggressive_scales, args.aggressive_scale),
                parse_float_list(args.drift_budgets, args.drift_budget),
                parse_float_list(args.protect_alphas, args.protect_alpha),
                parse_float_list(args.normal_alphas, args.normal_alpha),
                parse_float_list(args.aggressive_alphas, args.aggressive_alpha),
            )
        ]
    include_params = len(sweep_rows) > 1
    grids: List[Tuple[str, SchedulerConfig]] = []
    seen_names = set()
    def add_grid(name: str, config: SchedulerConfig) -> None:
        if name in seen_names:
            return
        seen_names.add(name)
        grids.append((name, config))

    if "fixed_delta_mixed" in requested_set:
        add_grid("fixed_delta_mixed", SchedulerConfig(**scalar_base))
    gate_policies = parse_str_list(getattr(args, "gate_policies", None), args.gate_policy)
    for row in sweep_rows:
        base = {
            **scalar_base,
            "high_start_threshold": row["high_start_threshold"],
            "blank_threshold": row["blank_threshold"],
            "low_margin_threshold": row["low_margin_threshold"],
            "high_margin_threshold": row["high_margin_threshold"],
            "protect_scale": row["protect_scale"],
            "aggressive_scale": row["aggressive_scale"],
            "drift_budget": row["drift_budget"],
            "protect_alpha": row["protect_alpha"],
            "normal_alpha": row["normal_alpha"],
            "aggressive_alpha": row["aggressive_alpha"],
            "budget_aggressive_ratio": row["budget_aggressive_ratio"],
            "budget_normal_ratio": row["budget_normal_ratio"],
            "budget_protect_ratio": row["budget_protect_ratio"],
            "precision_fake_quant": bool(int(row.get("precision_fake_quant", int(bool(args.precision_fake_quant))))),
            "precision_aggressive_bits": int(row["precision_aggressive_bits"]),
            "precision_normal_bits": int(row["precision_normal_bits"]),
            "precision_protect_bits": int(row["precision_protect_bits"]),
        }
        explicit_name = str(row.get("name") or "").strip()
        if "semantic_scheduler" in requested_set:
            config = SchedulerConfig(**base, enable_semantic=True)
            name = f"semantic_scheduler__{explicit_name}" if explicit_name else scheduler_name(
                "semantic_scheduler", config, include_params
            )
            add_grid(name, config)
        if "uniform_topk_delta" in requested_set:
            config = SchedulerConfig(**base, active_budget_policy="uniform")
            name = f"uniform_topk_delta__{explicit_name}" if explicit_name else scheduler_name(
                "uniform_topk_delta", config, include_params
            )
            add_grid(name, config)
        if "semantic_budget_topk" in requested_set:
            config = SchedulerConfig(**base, enable_semantic=True, active_budget_policy="semantic")
            name = f"semantic_budget_topk__{explicit_name}" if explicit_name else scheduler_name(
                "semantic_budget_topk", config, include_params
            )
            add_grid(name, config)
        if "blank_budget_topk" in requested_set:
            config = SchedulerConfig(
                **base,
                enable_semantic=True,
                signal_policy="blank_only",
                active_budget_policy="semantic",
            )
            name = f"blank_budget_topk__{explicit_name}" if explicit_name else scheduler_name(
                "blank_budget_topk", config, include_params
            )
            add_grid(name, config)
        if "margin_budget_topk" in requested_set:
            config = SchedulerConfig(
                **base,
                enable_semantic=True,
                signal_policy="margin_only",
                active_budget_policy="semantic",
            )
            name = f"margin_budget_topk__{explicit_name}" if explicit_name else scheduler_name(
                "margin_budget_topk", config, include_params
            )
            add_grid(name, config)
        if "semantic_precision_scheduler" in requested_set:
            config = SchedulerConfig(
                **{**base, "precision_fake_quant": False},
                enable_semantic=True,
                precision_policy="semantic",
            )
            name = f"semantic_precision_scheduler__{explicit_name}" if explicit_name else scheduler_name(
                "semantic_precision_scheduler", config, include_params
            )
            add_grid(name, config)
        if "semantic_precision_fake_quant" in requested_set:
            config = SchedulerConfig(
                **{**base, "precision_fake_quant": True},
                enable_semantic=True,
                precision_policy="semantic",
            )
            name = f"semantic_precision_fake_quant__{explicit_name}" if explicit_name else scheduler_name(
                "semantic_precision_fake_quant", config, include_params
            )
            add_grid(name, config)
        if "blank_only" in requested_set or "blank_only_scheduler" in requested_set:
            config = SchedulerConfig(**base, enable_semantic=True, signal_policy="blank_only")
            name = f"blank_only_scheduler__{explicit_name}" if explicit_name else scheduler_name(
                "blank_only_scheduler", config, include_params
            )
            add_grid(name, config)
        if "margin_only" in requested_set or "margin_only_scheduler" in requested_set:
            config = SchedulerConfig(**base, enable_semantic=True, signal_policy="margin_only")
            name = f"margin_only_scheduler__{explicit_name}" if explicit_name else scheduler_name(
                "margin_only_scheduler", config, include_params
            )
            add_grid(name, config)
        if "aggressive_fsm" in requested_set:
            aggressive_base = {**base, "aggressive_scale": max(float(base["aggressive_scale"]), 2.5)}
            config = SchedulerConfig(**aggressive_base, enable_semantic=True, signal_policy="aggressive_fsm")
            name = f"aggressive_fsm__{explicit_name}" if explicit_name else scheduler_name(
                "aggressive_fsm", config, include_params
            )
            add_grid(name, config)
        if "semantic_gate_scheduler" in requested_set:
            row_policy = str(row.get("gate_policy") or "").strip()
            policies = [row_policy] if row_policy else gate_policies
            for gate_policy in policies:
                config = SchedulerConfig(**base, enable_semantic=True, gate_policy=str(gate_policy))
                if explicit_name:
                    name = f"semantic_gate_scheduler__{explicit_name}"
                    if len(policies) > 1:
                        name = f"{name}__gp{gate_policy}"
                else:
                    name = scheduler_name("semantic_gate_scheduler", config, include_params or len(policies) > 1)
                add_grid(name, config)
        if "drift_bounded_delta" in requested_set:
            config = SchedulerConfig(**base, enable_drift_bound=True)
            name = f"drift_bounded_delta__{explicit_name}" if explicit_name else scheduler_name(
                "drift_bounded_delta", config, include_params
            )
            add_grid(name, config)
        if "semantic_drift_delta" in requested_set:
            config = SchedulerConfig(**base, enable_semantic=True, enable_drift_bound=True)
            name = f"semantic_drift_delta__{explicit_name}" if explicit_name else scheduler_name(
                "semantic_drift_delta", config, include_params
            )
            add_grid(name, config)
    if not grids:
        raise ValueError(f"No runnable variants requested from --variants={args.variants!r}")
    return grids


def run(args: argparse.Namespace) -> None:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if args.device == "auto":
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model_device = torch.device(args.device)
    manual_device = torch.device("cpu" if args.manual_device == "auto" else args.manual_device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PaperBrainToTextV5(**checkpoint["model_config"]).to(model_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = load_eval_loader(checkpoint, args, model_device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "workload_timestep_trace.csv"
    sample_path = output_dir / "semantic_scheduler_per_sample.csv"
    for path in [trace_path, sample_path, output_dir / "gate_active_trace.csv", output_dir / "active_set_trace.npz"]:
        if path.exists():
            path.unlink()

    cached = []
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
            session_names = batch.get(
                "session_names",
                [f"session_{int(session_ids[row].item())}" for row in range(inputs.shape[0])],
            )
            sentence_indices = batch.get(
                "sentence_indices",
                torch.arange(inputs.shape[0], dtype=torch.long),
            )
            block_nums = batch.get(
                "block_nums",
                torch.full((inputs.shape[0],), -1, dtype=torch.long),
            )
            for row in range(inputs.shape[0]):
                length = int(lengths[row].item())
                cached.append(
                    {
                        "inputs": inputs[row, :length].detach().cpu(),
                        "length": length,
                        "session_id": int(session_ids[row].item()),
                        "session_name": session_names[row],
                        "sentence_index": int(sentence_indices[row].item()),
                        "block_num": int(block_nums[row].item()),
                        "reference": batch["transcripts"][row],
                        "baseline_logits": logits[row, :length].detach().cpu(),
                        "char_start": char_starts[row, :length].detach().cpu(),
                        "sample_index": len(cached) + 1,
                    }
                )
            print(f"Cached baseline batch {batch_idx}/{len(loader)}", flush=True)

    model = model.to(manual_device)
    model.eval()
    variant = make_variant()
    schedulers = scheduler_grid(args)
    results: List[Dict[str, object]] = []
    all_trace_rows: List[Dict[str, object]] = []
    active_set_records: List[Dict[str, object]] = []

    for variant_name, sched in schedulers:
        hypotheses: List[str] = []
        references: List[str] = []
        aggregate_stats = HardwareStats()
        sample_rows: List[Dict[str, object]] = []
        for item in cached:
            trace_context = {
                "record_trace": len(all_trace_rows) < int(args.max_trace_rows),
                "variant": variant_name,
                "sample_index": item["sample_index"],
                "session_id": item["session_id"],
                "session_name": item["session_name"],
                "sentence_index": item["sentence_index"],
                "block_num": item["block_num"],
                "emit_active_sets": bool(args.emit_active_sets),
                "active_set_records": active_set_records,
                "max_active_set_records": int(args.max_active_set_records),
                "active_set_variant_filter": args.active_set_variant_filter,
            }
            logits, stats, trace_rows = semantic_forward_one(
                model,
                item["inputs"],
                item["length"],
                item["session_id"],
                variant,
                sched,
                manual_device,
                trace_context=trace_context,
            )
            if trace_rows:
                annotate_trace_events(
                    trace_rows,
                    item["reference"],
                    logits,
                    item["char_start"],
                    start_threshold=args.start_threshold,
                    start_delay=args.start_delay,
                    radius=args.near_start_radius,
                )
                all_trace_rows.extend(trace_rows)
                if len(all_trace_rows) >= int(args.max_trace_rows):
                    append_csv(trace_path, all_trace_rows)
                    all_trace_rows = []
            aggregate_stats.merge(stats)
            hyp = decode_paper_v5_outputs(
                logits,
                item["length"],
                start_threshold=float(args.start_threshold),
                start_delay=int(args.start_delay),
            )
            hypotheses.append(hyp)
            references.append(item["reference"])
            stat_summary = stats.summary()
            notes = ";".join(stats.notes)
            sample_rows.append(
                {
                    "variant": variant_name,
                    "sample_index": item["sample_index"],
                    "session_id": item["session_id"],
                    "session_name": item["session_name"],
                    "sentence_index": item["sentence_index"],
                    "block_num": item["block_num"],
                    "reference": render_transcript(item["reference"]),
                    "hypothesis": render_transcript(hyp),
                    **sample_error_counts(item["reference"], hyp),
                    **stat_summary,
                    "scheduler_notes": notes,
                }
            )
            print(f"{variant_name}: sample {item['sample_index']}/{len(cached)}", flush=True)
        append_csv(sample_path, sample_rows)
        metrics = compute_error_rates(references, hypotheses)
        row = {
            "variant": variant_name,
            "samples": len(cached),
            "cer": float(metrics["cer"]),
            "wer": float(metrics["wer"]),
            **aggregate_stats.summary(),
            "scheduler": json.dumps(sched.__dict__, sort_keys=True),
        }
        results.append(row)
        write_csv(output_dir / "semantic_scheduler_results.csv", results)

    if all_trace_rows:
        append_csv(trace_path, all_trace_rows)
    if trace_path.exists():
        with trace_path.open("r", newline="", encoding="utf-8") as handle:
            trace_rows = list(csv.DictReader(handle))
        write_csv(output_dir / "workload_trace_summary.csv", aggregate_trace(trace_rows))
    if args.emit_active_sets:
        save_active_set_outputs(output_dir, active_set_records)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved semantic characterization to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workload characterization and first semantic scheduler prototype for BCI GRU decoding."
    )
    parser.add_argument("--checkpoint", type=str, default="brain_to_text_v5_paper_best.pt")
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "Datasets"))
    parser.add_argument("--training_steps_root", type=str, default=None)
    parser.add_argument("--cv_part", type=str, default="HeldOutBlocks", choices=["HeldOutBlocks", "HeldOutTrials"])
    parser.add_argument("--label_cache", type=str, default="paper_v5_label_cache.pt")
    parser.add_argument("--use_existing_label_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_records_per_session", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--manual_device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--truncate_time_steps", type=int, default=None)
    parser.add_argument("--start_threshold", type=float, default=0.3)
    parser.add_argument("--start_delay", type=int, default=15)
    parser.add_argument("--near_start_radius", type=int, default=15)
    parser.add_argument(
        "--variants",
        type=str,
        default="fixed_delta_mixed,semantic_scheduler,drift_bounded_delta,semantic_drift_delta",
        help=(
            "Comma-separated subset: fixed_delta_mixed, semantic_scheduler, uniform_topk_delta, "
            "semantic_budget_topk, blank_budget_topk, margin_budget_topk, semantic_precision_scheduler, "
            "semantic_precision_fake_quant, blank_only_scheduler, margin_only_scheduler, aggressive_fsm, "
            "semantic_gate_scheduler, drift_bounded_delta, semantic_drift_delta."
        ),
    )
    parser.add_argument(
        "--gate_policy",
        type=str,
        default="semantic_zn",
        choices=[
            "semantic_zn",
            "reset_relaxed",
            "candidate_relaxed",
            "all_relaxed",
            "z_protected_n_relaxed",
            "rn_relaxed_z_protected",
            "all_mild",
            "aggressive_only",
        ],
        help="Gate-specific threshold policy used only by semantic_gate_scheduler.",
    )
    parser.add_argument(
        "--gate_policies",
        type=str,
        default=None,
        help=(
            "Comma-separated semantic_gate_scheduler policies. Recommended small matrix: "
            "candidate_relaxed,reset_relaxed,z_protected_n_relaxed,rn_relaxed_z_protected,all_mild,aggressive_only."
        ),
    )
    parser.add_argument("--base_input_threshold", type=float, default=0.08)
    parser.add_argument("--base_hidden_threshold", type=float, default=0.02)
    parser.add_argument("--protect_scale", type=float, default=0.5)
    parser.add_argument("--aggressive_scale", type=float, default=2.0)
    parser.add_argument("--high_start_threshold", type=float, default=0.3)
    parser.add_argument("--blank_threshold", type=float, default=0.08)
    parser.add_argument("--low_margin_threshold", type=float, default=1.0)
    parser.add_argument("--high_margin_threshold", type=float, default=2.5)
    parser.add_argument("--drift_budget", type=float, default=5000.0)
    parser.add_argument("--protect_alpha", type=float, default=512.0)
    parser.add_argument("--normal_alpha", type=float, default=2048.0)
    parser.add_argument("--aggressive_alpha", type=float, default=4096.0)
    parser.add_argument(
        "--drift_normalizer",
        type=str,
        default="raw",
        choices=["raw", "per_output", "sensitivity_mean", "preact_norm"],
    )
    parser.add_argument(
        "--budget_aggressive_ratio",
        type=float,
        default=0.20,
        help="Active-column budget ratio for AGGRESSIVE mode in top-K budget variants.",
    )
    parser.add_argument(
        "--budget_normal_ratio",
        type=float,
        default=0.50,
        help="Active-column budget ratio for NORMAL mode or uniform top-K variants.",
    )
    parser.add_argument(
        "--budget_protect_ratio",
        type=float,
        default=1.00,
        help="Active-column budget ratio for PROTECT mode in semantic top-K variants.",
    )
    parser.add_argument(
        "--precision_aggressive_bits",
        type=int,
        default=8,
        help="Effective datapath bits used for AGGRESSIVE mode in semantic precision BOP/traffic proxy.",
    )
    parser.add_argument(
        "--precision_fake_quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable numerical fake-quantization for semantic precision variants. The named semantic_precision_fake_quant variant enables this automatically.",
    )
    parser.add_argument(
        "--precision_normal_bits",
        type=int,
        default=12,
        help="Effective datapath bits used for NORMAL mode in semantic precision BOP/traffic proxy.",
    )
    parser.add_argument(
        "--precision_protect_bits",
        type=int,
        default=16,
        help="Effective datapath bits used for PROTECT mode in semantic precision BOP/traffic proxy.",
    )
    parser.add_argument("--protect_scales", type=str, default=None)
    parser.add_argument("--aggressive_scales", type=str, default=None)
    parser.add_argument("--high_start_thresholds", type=str, default=None)
    parser.add_argument("--blank_thresholds", type=str, default=None)
    parser.add_argument("--low_margin_thresholds", type=str, default=None)
    parser.add_argument("--high_margin_thresholds", type=str, default=None)
    parser.add_argument("--drift_budgets", type=str, default=None)
    parser.add_argument("--protect_alphas", type=str, default=None)
    parser.add_argument("--normal_alphas", type=str, default=None)
    parser.add_argument("--aggressive_alphas", type=str, default=None)
    parser.add_argument(
        "--scheduler_config_csv",
        type=str,
        default=None,
        help="Optional CSV with sparse scheduler candidates. Columns can include name, blank_threshold, high_margin_threshold, aggressive_scale, etc.",
    )
    parser.add_argument("--max_trace_rows", type=int, default=200000)
    parser.add_argument("--emit_active_sets", action="store_true")
    parser.add_argument("--active_set_format", type=str, default="bitmask", choices=["bitmask", "indices"])
    parser.add_argument("--phase_grouping", type=str, default="both", choices=["semantic", "fsm", "both"])
    parser.add_argument("--gate_locality_metrics", action="store_true")
    parser.add_argument("--memory_model", type=str, default="none", choices=["none", "active_set"])
    parser.add_argument("--mask_encoding", type=str, default="full,sparse,xor,rle")
    parser.add_argument("--active_column_cache_K", type=str, default="1,2,4,8,16")
    parser.add_argument("--bank_model", type=str, default="8,16,32")
    parser.add_argument("--max_active_set_records", type=int, default=1000000)
    parser.add_argument(
        "--active_set_variant_filter",
        type=str,
        default="semantic_gate_scheduler",
        help="Comma-separated substrings; active-set masks are emitted only for matching variant names.",
    )
    parser.add_argument("--affine_ranks", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--output_dir", type=str, default="artifacts/semantic_workload")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
