"""导出并验证 H144-v2 的纯整数、bit-exact FPGA golden model。

运行时矩阵乘加、移位、舍入、饱和、GRU hidden 回写和 sigmoid/tanh LUT
均使用整数实现；不调用 ``torch.nn.GRU``。浮点只用于离线参数量化、输入 ADC/特征
到 INT8 的边界转换，以及把最终 logits 反量化后计算软件指标。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from brain_to_text_main_v5 import compute_error_rates
from paper_v5_pipeline import (
    PaperV5EvalDataset,
    collate_variable_paper_batch,
    decode_paper_v5_outputs,
)
from recurrent_fixed_point_model import RecurrentFixedPointH144, symmetric_limits
from spatiotemporal_discriminability_atlas import split_train_internal
from train_compact_bci import load_frozen_label_cache


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def round_divide_even(values: torch.Tensor, denominator: int) -> torch.Tensor:
    """整数除法，采用 round-to-nearest ties-to-even。"""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))
    absolute = values.abs()
    quotient = torch.div(absolute, denominator, rounding_mode="floor")
    remainder = absolute - quotient * denominator
    twice = remainder * 2
    increment = (twice > denominator) | ((twice == denominator) & ((quotient & 1) == 1))
    return sign * (quotient + increment.to(quotient.dtype))


def round_shift_even(values: torch.Tensor, right_shift: int) -> torch.Tensor:
    if right_shift == 0:
        return values
    if right_shift < 0:
        return values << (-right_shift)
    return round_divide_even(values, 1 << right_shift)


def saturate(values: torch.Tensor, bits: int, signed: bool = True) -> torch.Tensor:
    if signed:
        qmin, qmax = symmetric_limits(bits)
    else:
        qmin, qmax = 0, (1 << bits) - 1
    return values.clamp(qmin, qmax).to(torch.int64)


def quantize_float(values: torch.Tensor, exponent: int, bits: int, signed: bool = True) -> torch.Tensor:
    scale = float(2.0**int(exponent))
    return saturate(torch.round(values.double() / scale).to(torch.int64), bits, signed)


def weight_integer_and_exponent(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qmin, qmax = symmetric_limits(8)
    reduce_dims = tuple(range(1, values.ndim))
    source = values.detach().float()
    maximum = source.abs().amax(dim=reduce_dims).clamp_min(1.0e-12)
    exponent = torch.ceil(torch.log2(maximum / float(qmax))).to(torch.int64)
    reshape = [values.shape[0]] + [1] * (values.ndim - 1)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponent.float()).reshape(reshape)
    integer = torch.round(source / scale).clamp(qmin, qmax).to(torch.int64)
    return integer, exponent


def requantize_per_output(
    values: torch.Tensor,
    source_exponent: torch.Tensor,
    target_exponent: int,
    bits: int,
    signed: bool = True,
) -> torch.Tensor:
    """对最后一个维度逐输出通道执行固定移位和饱和。"""

    shifts = int(target_exponent) - source_exponent.to(torch.int64)
    result = torch.empty_like(values, dtype=torch.int64)
    for shift in torch.unique(shifts).tolist():
        mask = shifts == int(shift)
        result[..., mask] = round_shift_even(values[..., mask], int(shift))
    return saturate(result, bits, signed)


@dataclass
class GoldenStats:
    accumulator_max_abs: Dict[str, int] = field(default_factory=dict)
    accumulator_overflow_count: Dict[str, int] = field(default_factory=dict)

    def record_accumulator(self, name: str, values: torch.Tensor) -> torch.Tensor:
        maximum = int(values.abs().max().item()) if values.numel() else 0
        self.accumulator_max_abs[name] = max(self.accumulator_max_abs.get(name, 0), maximum)
        overflow = int(((values < INT32_MIN) | (values > INT32_MAX)).sum().item())
        self.accumulator_overflow_count[name] = self.accumulator_overflow_count.get(name, 0) + overflow
        return values.clamp(INT32_MIN, INT32_MAX)


class IntegerLinear:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, name: str) -> None:
        self.weight, self.weight_exponent = weight_integer_and_exponent(weight.cpu())
        self.bias = None if bias is None else bias.detach().double().cpu()
        self.name = name

    def __call__(
        self,
        inputs: torch.Tensor,
        input_exponent: int,
        output_exponent: int,
        output_bits: int,
        stats: GoldenStats,
        signed: bool = True,
    ) -> torch.Tensor:
        accumulator = torch.matmul(inputs.to(torch.int64), self.weight.t())
        product_exponent = self.weight_exponent + int(input_exponent)
        if self.bias is not None:
            scale = torch.pow(torch.tensor(2.0, dtype=torch.float64), product_exponent.double())
            bias_integer = torch.round(self.bias / scale).clamp(INT32_MIN, INT32_MAX).to(torch.int64)
            accumulator = accumulator + bias_integer
        accumulator = stats.record_accumulator(self.name, accumulator)
        return requantize_per_output(
            accumulator,
            product_exponent,
            output_exponent,
            output_bits,
            signed,
        )


class IntegerLUT:
    def __init__(self, entries: int, value_range: float, preact_exponent: int) -> None:
        self.entries = int(entries)
        self.value_range = float(value_range)
        self.preact_exponent = int(preact_exponent)
        grid = torch.linspace(-value_range, value_range, entries, dtype=torch.float32)
        self.sigmoid = quantize_float(torch.sigmoid(grid), -16, 16, signed=False)
        self.tanh = quantize_float(torch.tanh(grid), -15, 16, signed=True)
        self.minimum_input = int(round(-value_range / (2.0**preact_exponent)))
        self.maximum_input = int(round(value_range / (2.0**preact_exponent)))
        self.denominator = self.maximum_input - self.minimum_input

    def index(self, values: torch.Tensor) -> torch.Tensor:
        clipped = values.clamp(self.minimum_input, self.maximum_input)
        numerator = (clipped - self.minimum_input) * (self.entries - 1)
        return round_divide_even(numerator, self.denominator).clamp(0, self.entries - 1).long()


class IntegerGRU:
    def __init__(
        self,
        gru: torch.nn.GRU,
        input_exponent: int,
        hidden_exponent: int,
        preact_exponent: int,
        start_state: torch.Tensor,
        lut: IntegerLUT,
        name: str,
    ) -> None:
        self.input_exponent = int(input_exponent)
        self.hidden_exponent = int(hidden_exponent)
        self.preact_exponent = int(preact_exponent)
        self.input_linear = IntegerLinear(gru.weight_ih_l0, gru.bias_ih_l0, f"{name}.input")
        self.hidden_linear = IntegerLinear(gru.weight_hh_l0, gru.bias_hh_l0, f"{name}.hidden")
        self.start_state = quantize_float(start_state.detach().cpu()[0, 0], hidden_exponent, 16)
        self.lut = lut
        self.name = name

    def __call__(self, inputs: torch.Tensor, stats: GoldenStats) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_gates = self.input_linear(
            inputs,
            self.input_exponent,
            self.preact_exponent,
            16,
            stats,
        )
        hidden = self.start_state.clone()
        outputs: List[torch.Tensor] = []
        resets: List[torch.Tensor] = []
        updates: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        hidden_linears: List[torch.Tensor] = []
        previous_hiddens: List[torch.Tensor] = []
        for time_index in range(inputs.shape[0]):
            previous_hiddens.append(hidden.clone())
            hidden_gates = self.hidden_linear(
                hidden,
                self.hidden_exponent,
                self.preact_exponent,
                16,
                stats,
            )
            hidden_linears.append(hidden_gates)
            input_reset, input_update, input_new = input_gates[time_index].chunk(3)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3)
            reset_pre = saturate(input_reset + hidden_reset, 16)
            update_pre = saturate(input_update + hidden_update, 16)
            reset = self.lut.sigmoid[self.lut.index(reset_pre)]
            update = self.lut.sigmoid[self.lut.index(update_pre)]
            reset_hidden_product = reset * hidden_new
            reset_hidden = saturate(
                round_shift_even(
                    reset_hidden_product,
                    self.preact_exponent - (-16 + self.preact_exponent),
                ),
                16,
            )
            candidate_pre = saturate(input_new + reset_hidden, 16)
            candidate = self.lut.tanh[self.lut.index(candidate_pre)]
            one_minus_update = (1 << 16) - update
            candidate_term = saturate(
                round_shift_even(
                    one_minus_update * candidate,
                    self.hidden_exponent - (-16 - 15),
                ),
                16,
            )
            previous_term = saturate(
                round_shift_even(
                    update * hidden,
                    self.hidden_exponent - (-16 + self.hidden_exponent),
                ),
                16,
            )
            hidden = saturate(candidate_term + previous_term, 16)
            outputs.append(hidden)
            resets.append(reset)
            updates.append(update)
            candidates.append(candidate)
        return torch.stack(outputs), {
            "input_linear": input_gates,
            "hidden_linear": torch.stack(hidden_linears),
            "previous_hidden": torch.stack(previous_hiddens),
            "reset": torch.stack(resets),
            "update": torch.stack(updates),
            "candidate": torch.stack(candidates),
        }


class IntegerH144Golden:
    def __init__(self, model: RecurrentFixedPointH144) -> None:
        self.model = model.cpu().eval()
        specifications = model.fixed_point_specification()["quantizers"]
        self.exponents = {
            name: int(specification["scale_exponent"])
            for name, specification in specifications.items()
        }
        self.input_exponent = self.exponents["input_q"]
        self.adapter_mid_exponent = self.exponents["adapter_mid_q"]
        self.adapter_out_exponent = self.exponents["adapter_out_q"]
        self.depthwise_exponent = self.exponents["depthwise_q"]
        self.bottleneck_exponent = self.exponents["bottleneck_q"]
        self.gru2_input_exponent = self.exponents["gru2_input_q"]
        self.hidden1_exponent = self.exponents["hidden1_q"]
        self.hidden2_exponent = self.exponents["hidden2_q"]
        self.preact1_exponent = self.exponents["preact1_q"]
        self.preact2_exponent = self.exponents["preact2_q"]
        self.logit_exponent = self.exponents["logit_q"]
        self.lut = IntegerLUT(model.lut_entries, model.lut_range, self.preact1_exponent)
        temporal = model.temporal_bottleneck
        self.depthwise_weight, self.depthwise_weight_exponent = weight_integer_and_exponent(
            temporal.depthwise.weight.cpu()
        )
        self.pointwise = IntegerLinear(
            temporal.pointwise.weight[:, :, 0], temporal.pointwise.bias, "temporal.pointwise"
        )
        self.gru1 = IntegerGRU(
            model.gru1,
            self.bottleneck_exponent,
            self.hidden1_exponent,
            self.preact1_exponent,
            model.start_state_1,
            self.lut,
            "gru1",
        )
        self.gru2 = IntegerGRU(
            model.gru2,
            self.gru2_input_exponent,
            self.hidden2_exponent,
            self.preact2_exponent,
            model.start_state_2,
            self.lut,
            "gru2",
        )
        self.readout = IntegerLinear(model.readout.weight, model.readout.bias, "readout")

    def _adapter(self, inputs: torch.Tensor, session_id: int, stats: GoldenStats) -> torch.Tensor:
        adapter = self.model.session_adapter
        v = IntegerLinear(adapter.v[session_id], None, "adapter.v")
        u = IntegerLinear(adapter.u[session_id], None, "adapter.u")
        middle = v(
            inputs,
            self.input_exponent,
            self.adapter_mid_exponent,
            8,
            stats,
        )
        low_rank = u(
            middle,
            self.adapter_mid_exponent,
            self.adapter_out_exponent,
            8,
            stats,
        )
        residual = requantize_per_output(
            inputs,
            torch.full((inputs.shape[-1],), self.input_exponent, dtype=torch.int64),
            self.adapter_out_exponent,
            8,
        )
        bias = quantize_float(adapter.bias[session_id].detach().cpu(), self.adapter_out_exponent, 8)
        return saturate(residual + low_rank + bias, 8)

    def _depthwise(self, inputs: torch.Tensor, stats: GoldenStats) -> torch.Tensor:
        kernel = self.depthwise_weight.shape[-1]
        if self.model.temporal_bottleneck.causal:
            before, after = kernel - 1, 0
        else:
            before = after = kernel // 2
        padded = torch.zeros(
            (inputs.shape[0] + before + after, inputs.shape[1]), dtype=torch.int64
        )
        padded[before : before + inputs.shape[0]] = inputs
        accumulator = torch.zeros_like(inputs, dtype=torch.int64)
        weights = self.depthwise_weight[:, 0, :]
        for tap in range(kernel):
            accumulator += padded[tap : tap + inputs.shape[0]] * weights[:, tap]
        accumulator = stats.record_accumulator("temporal.depthwise", accumulator)
        product_exponent = self.depthwise_weight_exponent + self.adapter_out_exponent
        return requantize_per_output(
            accumulator,
            product_exponent,
            self.depthwise_exponent,
            8,
        )

    def forward_integer(
        self,
        inputs_float: torch.Tensor,
        session_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        stats = GoldenStats()
        inputs = quantize_float(inputs_float.detach().cpu(), self.input_exponent, 8)
        adapted = self._adapter(inputs, session_id, stats)
        depthwise = self._depthwise(adapted, stats)
        bottleneck = self.pointwise(
            depthwise,
            self.depthwise_exponent,
            self.bottleneck_exponent,
            8,
            stats,
        )
        hidden1, trace1 = self.gru1(bottleneck, stats)
        downsampled_h = hidden1[:: self.model.skip_len]
        source_exponents = torch.full(
            (downsampled_h.shape[-1],), self.hidden1_exponent, dtype=torch.int64
        )
        gru2_inputs = requantize_per_output(
            downsampled_h,
            source_exponents,
            self.gru2_input_exponent,
            8,
        )
        hidden2, trace2 = self.gru2(gru2_inputs, stats)
        logits_down = self.readout(
            hidden2,
            self.hidden2_exponent,
            self.logit_exponent,
            16,
            stats,
        )
        logits = torch.repeat_interleave(logits_down, repeats=self.model.skip_len, dim=0)
        logits = logits[: inputs.shape[0]]
        return logits, {
            "input": inputs,
            "adapter": adapted,
            "depthwise": depthwise,
            "bottleneck": bottleneck,
            "gru1_hidden": hidden1,
            "gru2_input": gru2_inputs,
            "gru2_hidden": hidden2,
            "logits_down": logits_down,
            "gru1_gates": trace1,
            "gru2_gates": trace2,
            "accumulator_max_abs": stats.accumulator_max_abs,
            "accumulator_overflow_count": stats.accumulator_overflow_count,
        }

    def export(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for name, parameter in self.model.named_parameters():
            if "weight" in name or name.endswith(".u") or name.endswith(".v"):
                if (name.endswith(".u") or name.endswith(".v")) and parameter.ndim == 3:
                    integer_rows = []
                    exponent_rows = []
                    for session_weights in parameter.detach().cpu():
                        integer_session, exponent_session = weight_integer_and_exponent(session_weights)
                        integer_rows.append(integer_session)
                        exponent_rows.append(exponent_session)
                    integer = torch.stack(integer_rows)
                    exponent = torch.stack(exponent_rows)
                else:
                    integer, exponent = weight_integer_and_exponent(parameter.detach().cpu())
                arrays[name + ".integer"] = integer.numpy().astype(np.int8)
                arrays[name + ".scale_exponent"] = exponent.numpy().astype(np.int16)
            elif "bias" in name or "start_state" in name:
                arrays[name + ".float_source"] = parameter.detach().cpu().numpy().astype(np.float32)
        arrays["sigmoid_lut_u16"] = self.lut.sigmoid.numpy().astype(np.uint16)
        arrays["tanh_lut_i16"] = self.lut.tanh.numpy().astype(np.int16)
        np.savez_compressed(output_dir / "fpga_parameters.npz", **arrays)
        specification = self.model.fixed_point_specification()
        specification["integer_runtime"] = {
            "matrix_accumulator": "signed INT32 with saturation",
            "requantization": "power-of-two shift, round-to-nearest ties-to-even",
            "hidden_register": "signed INT16, written after every recurrent step",
            "nonlinear": f"{self.model.lut_entries}-entry LUT over [-{self.model.lut_range}, {self.model.lut_range}]",
        }
        (output_dir / "fixed_point_specification.json").write_text(
            json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def flatten_tensor_trace(trace: Dict[str, object], prefix: str = "") -> Dict[str, np.ndarray]:
    """递归展开整数 trace，便于 RTL 对拍到 GRU 门控级中间结果。"""

    arrays: Dict[str, np.ndarray] = {}
    for name, value in trace.items():
        full_name = f"{prefix}_{name}" if prefix else name
        if isinstance(value, torch.Tensor):
            arrays[full_name] = value.numpy()
        elif isinstance(value, dict):
            arrays.update(flatten_tensor_trace(value, full_name))
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--label_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_records", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RecurrentFixedPointH144(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    specification = checkpoint["fixed_point_specification"]
    mode = str(specification.get("activation_mode", "lut"))
    if mode != "lut":
        raise RuntimeError(f"Golden integer runtime requires LUT checkpoint, got {mode}")
    model.activation_mode = "lut"
    model.lut_entries = int(specification.get("lut_entries") or 256)
    model.lut_range = float(specification.get("lut_range", 8.0))
    model.quantization_enabled = True
    model.set_observer_mode(False, False)
    model.eval()
    golden = IntegerH144Golden(model)
    golden.export(output)

    payload = load_frozen_label_cache(args.label_cache, args.data_root)
    official_train = [record for record in payload["records"] if record.split == "train"]
    _fit, development, blocks = split_train_internal(official_train)
    dataset = PaperV5EvalDataset(payload["datasets_root"], development, 2, training_steps_root=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_variable_paper_batch)
    rows: List[Dict[str, object]] = []
    for row_index, batch in enumerate(loader):
        if args.max_records > 0 and row_index >= args.max_records:
            break
        length = int(batch["input_lengths"][0].item())
        inputs = batch["inputs"][0, :length]
        session_id = int(batch["session_ids"][0].item())
        integer_logits, trace = golden.forward_integer(inputs, session_id)
        dequantized = integer_logits.float() * float(2.0**golden.logit_exponent)
        with torch.no_grad():
            qat_logits, _ = model(
                inputs.unsqueeze(0),
                torch.tensor([length]),
                torch.tensor([session_id]),
            )
        qat_logits = qat_logits[0, :length]
        difference = qat_logits - dequantized
        metrics = checkpoint["validation_metrics"]
        integer_text = decode_paper_v5_outputs(
            dequantized,
            length,
            float(metrics["start_threshold"]),
            int(metrics["start_delay"]),
        )
        qat_text = decode_paper_v5_outputs(
            qat_logits,
            length,
            float(metrics["start_threshold"]),
            int(metrics["start_delay"]),
        )
        row = {
            "record": row_index,
            "session": batch["session_names"][0],
            "sentence_index": int(batch["sentence_indices"][0].item()),
            "length": length,
            "reference": batch["transcripts"][0],
            "qat_text": qat_text,
            "integer_text": integer_text,
            "decoded_match": qat_text == integer_text,
            "logit_max_abs_difference": float(difference.abs().max().item()),
            "logit_mean_abs_difference": float(difference.abs().mean().item()),
            "accumulator_max_abs": trace["accumulator_max_abs"],
            "accumulator_overflow_count": trace["accumulator_overflow_count"],
        }
        rows.append(row)
        if row_index == 0:
            np.savez_compressed(
                output / "golden_trace_record0.npz",
                **flatten_tensor_trace(trace),
            )
    integer_errors = compute_error_rates(
        [str(row["reference"]) for row in rows],
        [str(row["integer_text"]) for row in rows],
    )
    qat_errors = compute_error_rates(
        [str(row["reference"]) for row in rows],
        [str(row["qat_text"]) for row in rows],
    )
    summary = {
        "checkpoint": args.checkpoint,
        "protocol": {
            "development_records": len(development),
            "development_blocks": blocks,
            "verified_records": len(rows),
            "official_heldout_signals_loaded": False,
        },
        "all_decoded_match": all(bool(row["decoded_match"]) for row in rows),
        "integer_cer": float(integer_errors["cer"]),
        "integer_wer": float(integer_errors["wer"]),
        "qat_cer_on_verified_records": float(qat_errors["cer"]),
        "qat_wer_on_verified_records": float(qat_errors["wer"]),
        "max_logit_abs_difference": max(
            (float(row["logit_max_abs_difference"]) for row in rows), default=0.0
        ),
        "integer_overflow_total": sum(
            sum(int(value) for value in row["accumulator_overflow_count"].values())
            for row in rows
        ),
        "records": rows,
    }
    (output / "verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
