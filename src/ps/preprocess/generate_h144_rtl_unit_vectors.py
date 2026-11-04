#!/usr/bin/env python3
"""生成 H144 RTL requantizer 与 LUT 的确定性穷举/随机测试向量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_vivado_mem import TensorFormat, verify_mem_round_trip, write_mem


def round_shift_even(value: int, right_shift: int) -> int:
    if right_shift <= 0:
        return value << (-right_shift)
    if right_shift >= 63:
        return 0
    negative = value < 0
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << right_shift)
    half = 1 << (right_shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return -quotient if negative else quotient


def saturate_symmetric(value: int, bits: int) -> int:
    limit = (1 << (bits - 1)) - 1
    return min(max(value, -limit), limit)


def lut_index(preactivation: int) -> int:
    clipped = min(max(preactivation, -16384), 16384)
    numerator = (clipped + 16384) * 511
    return min(max(round_shift_even(numerator, 15), 0), 511)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260929)
    directed = np.asarray(
        [
            -(1 << 31),
            -(1 << 31) + 1,
            -65537,
            -65536,
            -65535,
            -32769,
            -32768,
            -32767,
            -257,
            -256,
            -255,
            -129,
            -128,
            -127,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            127,
            128,
            129,
            255,
            256,
            257,
            32767,
            32768,
            32769,
            65535,
            65536,
            65537,
            (1 << 31) - 2,
            (1 << 31) - 1,
        ],
        dtype=np.int64,
    )
    random_values = rng.integers(-(1 << 31), 1 << 31, args.random_cases, dtype=np.int64)
    accumulators = np.concatenate([directed, random_values]).astype(np.int32)
    shifts = rng.integers(-4, 33, accumulators.size, dtype=np.int16)
    # 定向覆盖 ties-to-even 的半整数边界。
    shifts[: directed.size] = np.resize(np.arange(0, 17, dtype=np.int16), directed.size)
    output_is_int16 = rng.integers(0, 2, accumulators.size, dtype=np.uint8)
    output_is_int16[: directed.size] = np.resize(np.asarray([0, 1], dtype=np.uint8), directed.size)

    expected = np.empty(accumulators.size, dtype=np.int16)
    for index, (value, shift, is_i16) in enumerate(
        zip(accumulators, shifts, output_is_int16, strict=True)
    ):
        rounded = round_shift_even(int(value), int(shift))
        bits = 16 if int(is_i16) else 8
        expected[index] = saturate_symmetric(rounded, bits)

    requant_arrays = {
        "requant_accumulator_i32": (accumulators, TensorFormat(32, True)),
        "requant_shift_i8": (shifts.astype(np.int8), TensorFormat(8, True)),
        "requant_output_is_i16": (output_is_int16, TensorFormat(8, False)),
        "requant_expected_i16": (expected, TensorFormat(16, True)),
    }

    with np.load(args.parameters, allow_pickle=False) as archive:
        sigmoid = archive["sigmoid_lut_u16"].astype(np.uint16)
        tanh = archive["tanh_lut_i16"].astype(np.int16)
    # 文件索引按 16-bit 原始码排列，使 testbench 可直接用 raw[15:0] 驱动。
    preactivation_values = np.arange(65536, dtype=np.uint16).view(np.int16).astype(np.int32)
    indices = np.asarray([lut_index(int(value)) for value in preactivation_values], dtype=np.uint16)
    lut_arrays = {
        "lut_expected_index_u16": (indices, TensorFormat(16, False)),
        "lut_expected_sigmoid_u16": (sigmoid[indices], TensorFormat(16, False)),
        "lut_expected_tanh_i16": (tanh[indices], TensorFormat(16, True)),
    }

    manifest = {"requant_cases": int(accumulators.size), "lut_cases": 65536, "files": {}}
    for name, (array, fmt) in {**requant_arrays, **lut_arrays}.items():
        path = args.output_dir / f"{name}.mem"
        write_mem(path, array, fmt)
        verify_mem_round_trip(path, array, fmt)
        manifest["files"][name] = {
            "file": path.name,
            "elements": int(array.size),
            "bits": fmt.bits,
            "signed": fmt.signed,
        }
    (args.output_dir / "unit_vector_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
