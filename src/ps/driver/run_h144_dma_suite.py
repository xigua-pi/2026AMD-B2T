#!/usr/bin/env python3
"""在 PYNQ-Z2 上逐条执行官方 HeldOutBlocks bit-exact DMA 测试。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from pynq import MMIO, Overlay, allocate

from run_h144_dma_test import PL_CLOCK_HZ, percentile, run_once


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitstream", default="h144_dma_overlay.bit")
    parser.add_argument("--trace-dir", default="official_traces")
    parser.add_argument("--manifest", default="official_traces/manifest.json")
    parser.add_argument("--output", default="h144_official_board_result.json")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    trace_dir = (here / args.trace_dir).resolve()
    manifest_path = (here / args.manifest).resolve()
    output_path = (here / args.output).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    if args.max_records > 0:
        records = records[: args.max_records]

    overlay = Overlay(str((here / args.bitstream).resolve()), download=True)
    dma = overlay.axi_dma_0
    accelerator = overlay.ip_dict.get("h144_accel_0", {})
    mmio = MMIO(
        int(accelerator.get("phys_addr", 0x40000000)),
        int(accelerator.get("addr_range", 0x1000)),
    )

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    total_input_frames = 0
    total_output_words = 0

    for suite_index, metadata in enumerate(records):
        trace_path = trace_dir / str(metadata["trace_file"])
        with np.load(trace_path, allow_pickle=False) as trace:
            input_reference = np.asarray(trace["input"], dtype=np.int8)
            golden_logits = np.asarray(trace["logits_down"], dtype=np.int16)

        input_buffer = allocate(shape=input_reference.shape, dtype=np.int8)
        output_buffer = allocate(shape=golden_logits.shape, dtype=np.int16)
        try:
            input_buffer[:] = input_reference
            input_buffer.flush()
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
            actual = np.asarray(output_buffer, dtype=np.int16).copy()
        finally:
            input_buffer.close()
            output_buffer.close()

        difference = actual.astype(np.int32) - golden_logits.astype(np.int32)
        mismatch_indices = np.argwhere(difference != 0)
        mismatch_count = int(mismatch_indices.shape[0])
        max_abs_error = int(np.max(np.abs(difference))) if difference.size else 0
        pl_ms = float(registers["total_cycle_count"]) / (PL_CLOCK_HZ / 1000.0)
        input_duration_ms = float(input_reference.shape[0]) * 20.0
        expected_counters = (
            registers["status"] & 0xF == 0x6
            and registers["error_flags"] == 0
            and registers["input_byte_count"] == input_reference.size
            and registers["input_frame_count"] == input_reference.shape[0]
            and registers["output_word_count"] == golden_logits.size
            and registers["frame_done_count"] == input_reference.shape[0] + 6
            and registers["result_frame_count"] == input_reference.shape[0]
            and registers["flush_completed_count"] == 6
        )
        row = {
            "record_index": int(metadata["record_index"]),
            "session": str(metadata["session"]),
            "session_id": int(metadata["session_id"]),
            "sentence_index": int(metadata["sentence_index"]),
            "block_num": int(metadata["block_num"]),
            "input_frames": int(input_reference.shape[0]),
            "output_words": int(golden_logits.size),
            "latency_ms": latency_ms,
            "pl_transaction_ms": pl_ms,
            "real_time_factor": latency_ms / input_duration_ms,
            "mismatch_count": mismatch_count,
            "max_abs_error": max_abs_error,
            "counter_checks_passed": bool(expected_counters),
            "status": int(registers["status"]),
            "error_flags": int(registers["error_flags"]),
            "max_frame_cycles": int(registers["max_frame_cycles"]),
        }
        if mismatch_count:
            first = mismatch_indices[0]
            row["first_mismatch"] = {
                "frame": int(first[0]),
                "class": int(first[1]),
                "expected": int(golden_logits[tuple(first)]),
                "actual": int(actual[tuple(first)]),
            }
        results.append(row)
        total_input_frames += int(input_reference.shape[0])
        total_output_words += int(golden_logits.size)

        latencies = [float(item["latency_ms"]) for item in results]
        pl_latencies = [float(item["pl_transaction_ms"]) for item in results]
        rtfs = [float(item["real_time_factor"]) for item in results]
        partial_result = {
            "schema_version": 1,
            "test": "official HeldOutBlocks PYNQ-Z2 DDR-DMA-H144-DMA-DDR suite",
            "manifest": manifest_path.name,
            "target_records": len(records),
            "completed_records": len(results),
            "elapsed_seconds": time.perf_counter() - started,
            "bit_exact_records": sum(
                int(item["mismatch_count"] == 0) for item in results
            ),
            "total_mismatches": sum(int(item["mismatch_count"]) for item in results),
            "all_counter_checks_passed": all(
                bool(item["counter_checks_passed"]) for item in results
            ),
            "latency_ms": {
                "mean": float(statistics.fmean(latencies)),
                "median": float(statistics.median(latencies)),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "minimum": min(latencies),
                "maximum": max(latencies),
            },
            "pl_transaction_ms": {
                "total": sum(pl_latencies),
                "mean": float(statistics.fmean(pl_latencies)),
            },
            "real_time_factor": {
                "mean_per_record": float(statistics.fmean(rtfs)),
                "maximum": max(rtfs),
                "aggregate": sum(latencies) / (total_input_frames * 20.0),
            },
            "throughput": {
                "total_input_frames": total_input_frames,
                "total_output_words": total_output_words,
                "input_frames_per_second_pl": total_input_frames
                / (sum(pl_latencies) / 1000.0),
            },
            "records": results,
            "complete": len(results) == len(records),
            "passed": len(results) == len(records)
            and all(int(item["mismatch_count"]) == 0 for item in results)
            and all(bool(item["counter_checks_passed"]) for item in results),
        }
        write_json_atomic(output_path, partial_result)
        print(
            f"board={suite_index + 1}/{len(records)} "
            f"frames={input_reference.shape[0]} latency_ms={latency_ms:.3f} "
            f"mismatches={mismatch_count} counters={expected_counters}",
            flush=True,
        )

    return 0 if partial_result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
