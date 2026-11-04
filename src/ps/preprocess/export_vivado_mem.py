#!/usr/bin/env python3
"""把 H144 FPGA golden 参数与 trace 导出为 Vivado ``$readmemh`` 文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class TensorFormat:
    bits: int
    signed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def round_ties_to_even(values: np.ndarray) -> np.ndarray:
    """NumPy rint 遵循 round-to-nearest ties-to-even。"""

    return np.rint(values)


def saturate(values: np.ndarray, bits: int, signed: bool) -> np.ndarray:
    if signed:
        # 与软件 golden 一致：有符号格式采用对称饱和，不使用最小负数编码。
        lower, upper = -((1 << (bits - 1)) - 1), (1 << (bits - 1)) - 1
    else:
        lower, upper = 0, (1 << bits) - 1
    return np.clip(values, lower, upper)


def quantize(values: np.ndarray, exponent: int, bits: int, signed: bool) -> np.ndarray:
    scaled = round_ties_to_even(values.astype(np.float64) / (2.0**exponent))
    return saturate(scaled, bits, signed)


def accumulator_bias(
    bias: np.ndarray,
    weight_exponents: np.ndarray,
    input_exponent: int,
) -> np.ndarray:
    """将浮点偏置转换为相应 MAC 累加域中的 INT32 常量。"""

    product_exponents = weight_exponents.astype(np.int64) + int(input_exponent)
    scales = np.power(2.0, product_exponents.astype(np.float64))
    integers = round_ties_to_even(bias.astype(np.float64) / scales)
    return np.clip(integers, INT32_MIN, INT32_MAX).astype(np.int32)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "__", name).strip("_")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_mem(path: Path, values: np.ndarray, fmt: TensorFormat) -> None:
    """每行写一个定宽二补码 word；多维数组按 C order 展平。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(values).reshape(-1, order="C")
    mask = (1 << fmt.bits) - 1
    digits = (fmt.bits + 3) // 4
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for value in flat:
            encoded = int(value) & mask
            stream.write(f"{encoded:0{digits}x}\n")


def verify_mem_round_trip(path: Path, values: np.ndarray, fmt: TensorFormat) -> None:
    """重新解析生成的 MEM，防止位宽、符号扩展或展平顺序出错。"""

    encoded = np.asarray(
        [int(line, 16) for line in path.read_text(encoding="ascii").splitlines()],
        dtype=np.uint64,
    )
    if fmt.signed:
        sign_threshold = 1 << (fmt.bits - 1)
        full_range = 1 << fmt.bits
        decoded = encoded.astype(np.int64)
        decoded[encoded >= sign_threshold] -= full_range
    else:
        decoded = encoded.astype(np.int64)
    expected = np.asarray(values).reshape(-1, order="C").astype(np.int64)
    if not np.array_equal(decoded, expected):
        differences = np.flatnonzero(decoded != expected)
        first = int(differences[0]) if differences.size else -1
        raise RuntimeError(
            f"MEM 回读不一致：{path}，首个差异索引={first}，"
            f"期望={expected[first]}，实际={decoded[first]}"
        )


def parameter_formats(name: str, array: np.ndarray) -> TensorFormat:
    if name.endswith(".integer"):
        return TensorFormat(8, True)
    if name.endswith(".scale_exponent"):
        return TensorFormat(16, True)
    if name == "sigmoid_lut_u16":
        return TensorFormat(16, False)
    if name == "tanh_lut_i16":
        return TensorFormat(16, True)
    if name.endswith(".bias_acc_i32"):
        return TensorFormat(32, True)
    if name.endswith(".bias_i8"):
        return TensorFormat(8, True)
    if name.endswith(".start_i16"):
        return TensorFormat(16, True)
    raise KeyError(f"没有定义参数格式：{name}, dtype={array.dtype}")


def trace_format(name: str) -> TensorFormat:
    eight_bit = {"input", "adapter", "depthwise", "bottleneck", "gru2_input"}
    unsigned_gate = {"gru1_gates_reset", "gru1_gates_update", "gru2_gates_reset", "gru2_gates_update"}
    sixteen_bit = {
        "gru1_hidden",
        "gru2_hidden",
        "logits_down",
        "gru1_gates_input_linear",
        "gru1_gates_hidden_linear",
        "gru1_gates_previous_hidden",
        "gru1_gates_candidate",
        "gru2_gates_input_linear",
        "gru2_gates_hidden_linear",
        "gru2_gates_previous_hidden",
        "gru2_gates_candidate",
    }
    if name in eight_bit:
        return TensorFormat(8, True)
    if name in unsigned_gate:
        return TensorFormat(16, False)
    if name in sixteen_bit:
        return TensorFormat(16, True)
    raise KeyError(f"没有定义 trace 格式：{name}")


def prepare_integer_parameters(
    source: Dict[str, np.ndarray],
    specification: dict,
) -> Dict[str, np.ndarray]:
    quantizers = specification["quantizers"]
    exponent = {name: int(item["scale_exponent"]) for name, item in quantizers.items()}
    result = {
        name: np.asarray(value)
        for name, value in source.items()
        if not name.endswith(".float_source")
    }

    result["start_state_1.start_i16"] = quantize(
        source["start_state_1.float_source"], exponent["hidden1_q"], 16, True
    ).astype(np.int16)
    result["start_state_2.start_i16"] = quantize(
        source["start_state_2.float_source"], exponent["hidden2_q"], 16, True
    ).astype(np.int16)
    result["session_adapter.bias.bias_i8"] = quantize(
        source["session_adapter.bias.float_source"], exponent["adapter_out_q"], 8, True
    ).astype(np.int8)

    bias_domains = {
        "temporal_bottleneck.pointwise": exponent["depthwise_q"],
        "gru1.ih": exponent["bottleneck_q"],
        "gru1.hh": exponent["hidden1_q"],
        "gru2.ih": exponent["gru2_input_q"],
        "gru2.hh": exponent["hidden2_q"],
        "readout": exponent["hidden2_q"],
    }
    bias_sources = {
        "temporal_bottleneck.pointwise": (
            "temporal_bottleneck.pointwise.bias.float_source",
            "temporal_bottleneck.pointwise.weight.scale_exponent",
        ),
        "gru1.ih": ("gru1.bias_ih_l0.float_source", "gru1.weight_ih_l0.scale_exponent"),
        "gru1.hh": ("gru1.bias_hh_l0.float_source", "gru1.weight_hh_l0.scale_exponent"),
        "gru2.ih": ("gru2.bias_ih_l0.float_source", "gru2.weight_ih_l0.scale_exponent"),
        "gru2.hh": ("gru2.bias_hh_l0.float_source", "gru2.weight_hh_l0.scale_exponent"),
        "readout": ("readout.bias.float_source", "readout.weight.scale_exponent"),
    }
    for target, (bias_name, weight_exp_name) in bias_sources.items():
        result[target + ".bias_acc_i32"] = accumulator_bias(
            source[bias_name], source[weight_exp_name], bias_domains[target]
        )
    return result


def export_group(
    group_name: str,
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    format_resolver,
) -> dict:
    manifest: dict = {}
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        fmt = format_resolver(name, array) if group_name == "parameters" else format_resolver(name)
        path = group_dir / f"{safe_name(name)}.mem"
        write_mem(path, array, fmt)
        verify_mem_round_trip(path, array, fmt)
        manifest[name] = {
            "file": path.relative_to(output_dir).as_posix(),
            "shape": list(array.shape),
            "elements": int(array.size),
            "flatten_order": "C (last dimension changes fastest)",
            "bits": fmt.bits,
            "signed": fmt.signed,
            "source_dtype": str(array.dtype),
            "minimum": int(array.min()) if array.size else None,
            "maximum": int(array.max()) if array.size else None,
            "sha256": file_sha256(path),
        }
    return manifest


def write_systemverilog_package(output_dir: Path, specification: dict) -> None:
    q = specification["quantizers"]
    lines = [
        "`timescale 1ns/1ps",
        "",
        "package h144_fixed_pkg;",
        "  // Auto-generated from fixed_point_specification.json.",
        "  localparam int INPUT_DIM = 192;",
        "  localparam int BOTTLENECK_DIM = 80;",
        "  localparam int HIDDEN_DIM = 144;",
        "  localparam int OUTPUT_DIM = 32;",
        "  localparam int ADAPTER_RANK = 8;",
        "  localparam int TEMPORAL_KERNEL = 13;",
        "  localparam int SKIP_LEN = 5;",
        f"  localparam int LUT_ENTRIES = {int(specification['lut_entries'])};",
        f"  localparam int ACC_BITS = {int(specification['accumulator_bits'])};",
        f"  localparam int WEIGHT_BITS = {int(specification['weight_bits'])};",
    ]
    for name, item in q.items():
        constant = safe_name(name).upper() + "_EXP"
        lines.append(f"  localparam int signed {constant} = {int(item['scale_exponent'])};")
    lines.extend(["endpackage", ""])
    (output_dir / "h144_fixed_pkg.sv").write_text("\n".join(lines), encoding="ascii")


def pack_int8_rows(weights: np.ndarray) -> np.ndarray:
    """每 8 个 INT8 权重打包为一个 64-bit ROM word，lane 0 位于最低字节。"""

    rows = np.asarray(weights, dtype=np.int8)
    if rows.ndim != 2:
        raise ValueError(f"权重必须为二维矩阵，实际 shape={rows.shape}")
    padded_inputs = ((rows.shape[1] + 7) // 8) * 8
    padded = np.zeros((rows.shape[0], padded_inputs), dtype=np.int8)
    padded[:, : rows.shape[1]] = rows
    chunks = padded.reshape(rows.shape[0], -1, 8)
    packed = np.zeros(chunks.shape[:2], dtype=np.uint64)
    for lane in range(8):
        lane_values = chunks[:, :, lane].astype(np.uint8).astype(np.uint64)
        packed |= lane_values << np.uint64(8 * lane)
    return packed


def write_accelerator_memories(
    output_dir: Path,
    parameters: Dict[str, np.ndarray],
) -> dict:
    """构建供共享 8-lane MAC 引擎使用的统一 ROM 和布局常量。"""

    accelerator_dir = output_dir / "accelerator"
    accelerator_dir.mkdir(parents=True, exist_ok=True)

    adapter_v = parameters["session_adapter.v.integer"].reshape(-1, 192)
    adapter_u = parameters["session_adapter.u.integer"].reshape(-1, 8)
    adapter_v_exp = parameters["session_adapter.v.scale_exponent"].reshape(-1)
    adapter_u_exp = parameters["session_adapter.u.scale_exponent"].reshape(-1)

    operations = [
        {
            "name": "adapter_v",
            "weights": adapter_v,
            "exponents": adapter_v_exp,
            "bias": np.zeros(adapter_v.shape[0], dtype=np.int32),
            "sessions": 10,
            "outputs_per_session": 8,
        },
        {
            "name": "adapter_u",
            "weights": adapter_u,
            "exponents": adapter_u_exp,
            "bias": np.zeros(adapter_u.shape[0], dtype=np.int32),
            "sessions": 10,
            "outputs_per_session": 192,
        },
        {
            "name": "depthwise",
            "weights": parameters["temporal_bottleneck.depthwise.weight.integer"][:, 0, :],
            "exponents": parameters[
                "temporal_bottleneck.depthwise.weight.scale_exponent"
            ],
            "bias": np.zeros(192, dtype=np.int32),
            "sessions": 1,
            "outputs_per_session": 192,
        },
        {
            "name": "pointwise",
            "weights": parameters["temporal_bottleneck.pointwise.weight.integer"][:, :, 0],
            "exponents": parameters[
                "temporal_bottleneck.pointwise.weight.scale_exponent"
            ],
            "bias": parameters["temporal_bottleneck.pointwise.bias_acc_i32"],
            "sessions": 1,
            "outputs_per_session": 80,
        },
        {
            "name": "gru1_ih",
            "weights": parameters["gru1.weight_ih_l0.integer"],
            "exponents": parameters["gru1.weight_ih_l0.scale_exponent"],
            "bias": parameters["gru1.ih.bias_acc_i32"],
            "sessions": 1,
            "outputs_per_session": 432,
        },
        {
            "name": "gru1_hh",
            "weights": parameters["gru1.weight_hh_l0.integer"],
            "exponents": parameters["gru1.weight_hh_l0.scale_exponent"],
            "bias": parameters["gru1.hh.bias_acc_i32"],
            "sessions": 1,
            "outputs_per_session": 432,
        },
        {
            "name": "gru2_ih",
            "weights": parameters["gru2.weight_ih_l0.integer"],
            "exponents": parameters["gru2.weight_ih_l0.scale_exponent"],
            "bias": parameters["gru2.ih.bias_acc_i32"],
            "sessions": 1,
            "outputs_per_session": 432,
        },
        {
            "name": "gru2_hh",
            "weights": parameters["gru2.weight_hh_l0.integer"],
            "exponents": parameters["gru2.weight_hh_l0.scale_exponent"],
            "bias": parameters["gru2.hh.bias_acc_i32"],
            "sessions": 1,
            "outputs_per_session": 432,
        },
        {
            "name": "readout",
            "weights": parameters["readout.weight.integer"],
            "exponents": parameters["readout.weight.scale_exponent"],
            "bias": parameters["readout.bias_acc_i32"],
            "sessions": 1,
            "outputs_per_session": 32,
        },
    ]

    weight_words = []
    exponent_rows = []
    bias_rows = []
    layout: dict = {}
    weight_base = 0
    parameter_base = 0
    for operation in operations:
        weights = np.asarray(operation["weights"], dtype=np.int8)
        packed = pack_int8_rows(weights)
        exponents = np.asarray(operation["exponents"], dtype=np.int16).reshape(-1)
        biases = np.asarray(operation["bias"], dtype=np.int32).reshape(-1)
        if weights.shape[0] != exponents.size or weights.shape[0] != biases.size:
            raise ValueError(f"{operation['name']} 的权重、指数和偏置行数不一致")

        sessions = int(operation["sessions"])
        outputs_per_session = int(operation["outputs_per_session"])
        chunks_per_output = int(packed.shape[1])
        layout[operation["name"]] = {
            "weight_word_base": weight_base,
            "parameter_base": parameter_base,
            "input_length": int(weights.shape[1]),
            "padded_input_length": chunks_per_output * 8,
            "chunks_per_output": chunks_per_output,
            "total_output_rows": int(weights.shape[0]),
            "sessions": sessions,
            "outputs_per_session": outputs_per_session,
            "session_weight_word_stride": outputs_per_session * chunks_per_output,
            "session_parameter_stride": outputs_per_session,
        }
        weight_words.append(packed.reshape(-1))
        exponent_rows.append(exponents)
        bias_rows.append(biases)
        weight_base += int(packed.size)
        parameter_base += int(exponents.size)

    all_weights = np.concatenate(weight_words).astype(np.uint64)
    all_exponents = np.concatenate(exponent_rows).astype(np.int16)
    all_biases = np.concatenate(bias_rows).astype(np.int32)

    # 7-series BRAM 的深度为二次幂。若把 34,368 个 64-bit word 放进单个
    # 数组，Vivado 会按 65,536 深度映射，白白占用一半 BRAM。这里按
    # 16K + 16K + 2K 拆成三个独立 bank；第三个 bank 末尾补零。
    weight_bank_depths = (16384, 16384, 2048)
    if all_weights.size > sum(weight_bank_depths):
        raise ValueError(
            f"权重 word 数 {all_weights.size} 超过 bank 容量 {sum(weight_bank_depths)}"
        )
    padded_weights = np.zeros(sum(weight_bank_depths), dtype=np.uint64)
    padded_weights[: all_weights.size] = all_weights
    bank0_end = weight_bank_depths[0]
    bank1_end = bank0_end + weight_bank_depths[1]
    weight_banks = (
        padded_weights[:bank0_end],
        padded_weights[bank0_end:bank1_end],
        padded_weights[bank1_end:],
    )

    files = {
        "dense_weights_u64": (all_weights, TensorFormat(64, False)),
        "dense_weights_bank0_u64": (weight_banks[0], TensorFormat(64, False)),
        "dense_weights_bank1_u64": (weight_banks[1], TensorFormat(64, False)),
        "dense_weights_bank2_u64": (weight_banks[2], TensorFormat(64, False)),
        "dense_exponents_i16": (all_exponents, TensorFormat(16, True)),
        "dense_bias_i32": (all_biases, TensorFormat(32, True)),
        "adapter_bias_i8": (
            parameters["session_adapter.bias.bias_i8"],
            TensorFormat(8, True),
        ),
        "start_state_1_i16": (parameters["start_state_1.start_i16"], TensorFormat(16, True)),
        "start_state_2_i16": (parameters["start_state_2.start_i16"], TensorFormat(16, True)),
        "sigmoid_lut_u16": (parameters["sigmoid_lut_u16"], TensorFormat(16, False)),
        "tanh_lut_i16": (parameters["tanh_lut_i16"], TensorFormat(16, True)),
    }
    file_manifest = {}
    for name, (array, fmt) in files.items():
        path = accelerator_dir / f"{name}.mem"
        write_mem(path, array, fmt)
        verify_mem_round_trip(path, array, fmt)
        file_manifest[name] = {
            "file": path.relative_to(output_dir).as_posix(),
            "shape": list(np.asarray(array).shape),
            "elements": int(np.asarray(array).size),
            "bits": fmt.bits,
            "signed": fmt.signed,
            "sha256": file_sha256(path),
        }

    package_lines = [
        "`timescale 1ns/1ps",
        "",
        "package h144_layout_pkg;",
        "  // Auto-generated packed-ROM layout for the shared 8-lane MAC engine.",
        f"  localparam int DENSE_WEIGHT_WORDS = {all_weights.size};",
        f"  localparam int DENSE_WEIGHT_BANK0_WORDS = {weight_bank_depths[0]};",
        f"  localparam int DENSE_WEIGHT_BANK1_WORDS = {weight_bank_depths[1]};",
        f"  localparam int DENSE_WEIGHT_BANK2_WORDS = {weight_bank_depths[2]};",
        f"  localparam int DENSE_WEIGHT_BANK1_BASE = {weight_bank_depths[0]};",
        f"  localparam int DENSE_WEIGHT_BANK2_BASE = {bank1_end};",
        f"  localparam int DENSE_PARAMETER_ROWS = {all_exponents.size};",
    ]
    for name, item in layout.items():
        prefix = safe_name(name).upper()
        for key, value in item.items():
            constant = safe_name(key).upper()
            package_lines.append(f"  localparam int {prefix}_{constant} = {int(value)};")
    package_lines.extend(["endpackage", ""])
    (output_dir / "h144_layout_pkg.sv").write_text(
        "\n".join(package_lines), encoding="ascii"
    )

    accelerator_manifest = {
        "weight_word_lane_order": "lane 0 is bits [7:0], lane 7 is bits [63:56]",
        "row_order": "output-major; each output row contains contiguous input chunks",
        "layout": layout,
        "files": file_manifest,
    }
    (accelerator_dir / "accelerator_manifest.json").write_text(
        json.dumps(accelerator_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return accelerator_manifest


def write_readme(output_dir: Path, manifest: dict) -> None:
    text = f"""# H144 Vivado 仿真向量

本目录由 `tools/export_vivado_mem.py` 自动生成。

- 参数文件：`parameters/*.mem`
- record 0 输入和逐层期望值：`trace_record0/*.mem`
- 定点常量：`h144_fixed_pkg.sv`
- 完整形状、位宽、符号和哈希：`vivado_manifest.json`

全部 `.mem` 均为每行一个十六进制二补码 word，多维张量按 NumPy C order
展平，即最后一维变化最快。矩阵权重因此采用 `[output][input]` 顺序。

建议按照以下顺序逐级对拍：

1. input -> session adapter
2. depthwise temporal convolution
3. pointwise bottleneck
4. GRU1 gate 与 hidden
5. 5 倍降采样后的 GRU2 input、gate 与 hidden
6. readout logits

仿真必须实现 ties-to-even 舍入和对称饱和；若直接使用算术右移，负数及半整数边界
会与 golden model 不一致。

文件数：参数 {len(manifest['parameters'])} 个，trace {len(manifest['trace_record0'])} 个。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specification = json.loads(args.specification.read_text(encoding="utf-8"))
    with np.load(args.parameters, allow_pickle=False) as archive:
        source_parameters = {name: archive[name] for name in archive.files}
    with np.load(args.trace, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}

    integer_parameters = prepare_integer_parameters(source_parameters, specification)
    np.savez_compressed(args.output_dir / "rtl_integer_parameters.npz", **integer_parameters)

    manifest = {
        "format_version": 1,
        "rounding": specification["rounding"],
        "saturation": specification["saturation"],
        "source": {
            "parameters": str(args.parameters.resolve()),
            "trace": str(args.trace.resolve()),
            "specification": str(args.specification.resolve()),
        },
        "parameters": export_group(
            "parameters", integer_parameters, args.output_dir, parameter_formats
        ),
        "trace_record0": export_group(
            "trace_record0", trace, args.output_dir, trace_format
        ),
    }
    write_systemverilog_package(args.output_dir, specification)
    manifest["accelerator"] = write_accelerator_memories(
        args.output_dir, integer_parameters
    )
    manifest_path = args.output_dir / "vivado_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(args.output_dir, manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "parameter_mem_files": len(manifest["parameters"]),
                "trace_mem_files": len(manifest["trace_record0"]),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
