#!/usr/bin/env python3
"""为外接功率计提供同 bitstream 的 idle/active 可重复工作负载。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pynq import MMIO, Overlay, allocate

from run_h144_dma_test import run_once


XADC = Path("/sys/bus/iio/devices/iio:device0")


def read_temperature_celsius() -> float | None:
    """读取 Zynq XADC 温度；失败时不影响功率计工作负载。"""
    try:
        raw = float((XADC / "in_temp0_raw").read_text().strip())
        offset = float((XADC / "in_temp0_offset").read_text().strip())
        scale = float((XADC / "in_temp0_scale").read_text().strip())
        return (raw + offset) * scale / 1000.0
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("idle", "active"), required=True)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--record", type=int, default=98)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default="power_workload_result.json")
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be positive")

    here = Path(__file__).resolve().parent
    trace_dir = here / "official_traces"
    manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["records"][args.record]
    with np.load(trace_dir / metadata["trace_file"], allow_pickle=False) as trace:
        input_reference = np.asarray(trace["input"], dtype=np.int8)
        golden_logits = np.asarray(trace["logits_down"], dtype=np.int16)

    overlay = Overlay(str(here / "h144_dma_overlay.bit"), download=True)
    dma = overlay.axi_dma_0
    description = overlay.ip_dict.get("h144_accel_0", {})
    mmio = MMIO(
        int(description.get("phys_addr", 0x40000000)),
        int(description.get("addr_range", 0x1000)),
    )

    input_buffer = allocate(shape=input_reference.shape, dtype=np.int8)
    output_buffer = allocate(shape=golden_logits.shape, dtype=np.int16)
    try:
        input_buffer[:] = input_reference
        input_buffer.flush()
        start_temperature = read_temperature_celsius()
        started = time.monotonic()
        deadline = started + args.duration
        iterations = 0
        accelerator_ms = 0.0
        mismatch_count = 0
        error_flags_or = 0
        temperature_samples: list[dict[str, float]] = []
        next_temperature_sample = started

        if args.mode == "idle":
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_temperature_sample:
                    temperature = read_temperature_celsius()
                    if temperature is not None:
                        temperature_samples.append(
                            {"elapsed_seconds": now - started, "celsius": temperature}
                        )
                    next_temperature_sample = now + 1.0
                time.sleep(0.05)
        else:
            while time.monotonic() < deadline:
                latency_ms, registers = run_once(
                    dma=dma,
                    mmio=mmio,
                    input_buffer=input_buffer,
                    output_buffer=output_buffer,
                    input_frames=input_reference.shape[0],
                    output_words=golden_logits.size,
                    session_id=int(metadata["session_id"]),
                    timeout_s=args.timeout,
                )
                actual = np.asarray(output_buffer, dtype=np.int16)
                mismatch_count += int(np.count_nonzero(actual != golden_logits))
                error_flags_or |= int(registers["error_flags"])
                accelerator_ms += latency_ms
                iterations += 1
                now = time.monotonic()
                if now >= next_temperature_sample:
                    temperature = read_temperature_celsius()
                    if temperature is not None:
                        temperature_samples.append(
                            {"elapsed_seconds": now - started, "celsius": temperature}
                        )
                    next_temperature_sample = now + 1.0

        elapsed = time.monotonic() - started
        end_temperature = read_temperature_celsius()
        result = {
            "schema_version": 1,
            "mode": args.mode,
            "requested_duration_seconds": args.duration,
            "elapsed_seconds": elapsed,
            "bitstream": "h144_dma_overlay.bit",
            "clock_mhz": 50.0,
            "record_index": args.record,
            "input_frames_per_iteration": int(input_reference.shape[0]),
            "iterations": iterations,
            "accelerator_transaction_seconds": accelerator_ms / 1000.0,
            "workload_duty_cycle": accelerator_ms / (elapsed * 1000.0),
            "total_input_frames": iterations * int(input_reference.shape[0]),
            "total_mismatches": mismatch_count,
            "error_flags_or": error_flags_or,
            "temperature_start_celsius": start_temperature,
            "temperature_end_celsius": end_temperature,
            "temperature_samples": temperature_samples,
            "measurement_boundary": (
                "No power value is reported by this script. Read average input power "
                "from an external meter and pair it with this workload JSON."
            ),
            "passed": mismatch_count == 0 and error_flags_or == 0,
        }
        (here / args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 2
    finally:
        input_buffer.close()
        output_buffer.close()


if __name__ == "__main__":
    raise SystemExit(main())
