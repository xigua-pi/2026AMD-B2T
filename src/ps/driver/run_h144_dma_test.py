#!/usr/bin/env python3
"""PYNQ-Z2 上的 H144 DDR-DMA-PL-DDR 逐位对拍与时延测试。"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from pynq import MMIO, Overlay, allocate


REG_CONTROL = 0x00
REG_STATUS = 0x04
REG_SESSION_ID = 0x08
REG_INPUT_FRAMES = 0x0C
REG_OUTPUT_WORDS = 0x10
REG_INPUT_BYTE_COUNT = 0x14
REG_OUTPUT_WORD_COUNT = 0x18
REG_INPUT_FRAME_COUNT = 0x1C
REG_FRAME_DONE_COUNT = 0x20
REG_RESULT_FRAME_COUNT = 0x24
REG_FLUSH_COMPLETED_COUNT = 0x28
REG_TOTAL_CYCLE_COUNT = 0x2C
REG_MAX_FRAME_CYCLES = 0x30
REG_ERROR_FLAGS = 0x34
REG_DESIGN_VERSION = 0x38
REG_ENGINE_STATE = 0x3C

PL_CLOCK_HZ = 50_000_000


def percentile(values: list[float], q: float) -> float:
    """使用线性插值计算百分位数，避免依赖较新的 NumPy 参数名。"""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def read_registers(mmio: MMIO) -> dict[str, int]:
    """读取一次完整的控制器状态快照。"""
    return {
        "status": int(mmio.read(REG_STATUS)),
        "session_id": int(mmio.read(REG_SESSION_ID)),
        "input_frames_configured": int(mmio.read(REG_INPUT_FRAMES)),
        "output_words_configured": int(mmio.read(REG_OUTPUT_WORDS)),
        "input_byte_count": int(mmio.read(REG_INPUT_BYTE_COUNT)),
        "output_word_count": int(mmio.read(REG_OUTPUT_WORD_COUNT)),
        "input_frame_count": int(mmio.read(REG_INPUT_FRAME_COUNT)),
        "frame_done_count": int(mmio.read(REG_FRAME_DONE_COUNT)),
        "result_frame_count": int(mmio.read(REG_RESULT_FRAME_COUNT)),
        "flush_completed_count": int(mmio.read(REG_FLUSH_COMPLETED_COUNT)),
        "total_cycle_count": int(mmio.read(REG_TOTAL_CYCLE_COUNT)),
        "max_frame_cycles": int(mmio.read(REG_MAX_FRAME_CYCLES)),
        "error_flags": int(mmio.read(REG_ERROR_FLAGS)),
        "design_version": int(mmio.read(REG_DESIGN_VERSION)),
        "engine_state": int(mmio.read(REG_ENGINE_STATE)),
    }


def wait_transaction(dma, mmio: MMIO, timeout_s: float) -> None:
    """轮询 DMA 和加速器，超时后返回可诊断的寄存器信息。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = int(mmio.read(REG_STATUS))
        accelerator_done = bool(status & 0x2)
        if dma.sendchannel.idle and dma.recvchannel.idle and accelerator_done:
            # wait() 会执行 DMA 错误检查，并为接收缓冲区做 cache invalidate。
            dma.sendchannel.wait()
            dma.recvchannel.wait()
            return
        time.sleep(0.001)

    snapshot = read_registers(mmio)
    raise TimeoutError(
        "DMA/accelerator transaction timed out; "
        f"send_idle={dma.sendchannel.idle}, recv_idle={dma.recvchannel.idle}, "
        f"registers={snapshot}"
    )


def run_once(
    dma,
    mmio: MMIO,
    input_buffer,
    output_buffer,
    input_frames: int,
    output_words: int,
    session_id: int,
    timeout_s: float,
) -> tuple[float, dict[str, int]]:
    """完成一次 DDR->DMA->H144->DMA->DDR 的整句事务。"""
    output_buffer[:] = 0
    output_buffer.flush()

    mmio.write(REG_SESSION_ID, session_id)
    mmio.write(REG_INPUT_FRAMES, input_frames)
    mmio.write(REG_OUTPUT_WORDS, output_words)

    if int(mmio.read(REG_SESSION_ID)) != session_id:
        raise RuntimeError("AXI-Lite session_id readback mismatch")
    if int(mmio.read(REG_INPUT_FRAMES)) != input_frames:
        raise RuntimeError("AXI-Lite input_frames readback mismatch")
    if int(mmio.read(REG_OUTPUT_WORDS)) != output_words:
        raise RuntimeError("AXI-Lite output_words readback mismatch")

    # S2MM 必须先进入等待状态，否则首个输出可能遭遇不必要的反压。
    dma.recvchannel.transfer(output_buffer)
    mmio.write(REG_CONTROL, 1)
    start_ns = time.perf_counter_ns()
    dma.sendchannel.transfer(input_buffer)
    wait_transaction(dma, mmio, timeout_s)
    end_ns = time.perf_counter_ns()

    return (end_ns - start_ns) / 1_000_000.0, read_registers(mmio)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitstream", default="h144_dma_overlay.bit")
    parser.add_argument("--trace", default="golden_trace_record0.npz")
    parser.add_argument("--output", default="h144_dma_board_result.json")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--session-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    bitstream_path = (here / args.bitstream).resolve()
    trace_path = (here / args.trace).resolve()
    output_path = (here / args.output).resolve()

    with np.load(trace_path, allow_pickle=False) as trace:
        input_reference = np.asarray(trace["input"], dtype=np.int8)
        golden_logits = np.asarray(trace["logits_down"], dtype=np.int16)

    if input_reference.ndim != 2 or input_reference.shape[1] != 192:
        raise ValueError(f"unexpected input shape: {input_reference.shape}")
    if golden_logits.ndim != 2 or golden_logits.shape[1] != 32:
        raise ValueError(f"unexpected logits shape: {golden_logits.shape}")

    overlay = Overlay(str(bitstream_path), download=True)
    if "axi_dma_0" not in overlay.ip_dict:
        raise RuntimeError(f"axi_dma_0 missing from overlay: {overlay.ip_dict.keys()}")
    dma = overlay.axi_dma_0

    accelerator_description = overlay.ip_dict.get("h144_accel_0", {})
    accelerator_base = int(accelerator_description.get("phys_addr", 0x40000000))
    accelerator_range = int(accelerator_description.get("addr_range", 0x1000))
    mmio = MMIO(accelerator_base, accelerator_range)

    input_buffer = allocate(shape=input_reference.shape, dtype=np.int8)
    output_buffer = allocate(shape=golden_logits.shape, dtype=np.int16)
    try:
        input_buffer[:] = input_reference
        input_buffer.flush()

        latencies_ms: list[float] = []
        run_registers: list[dict[str, int]] = []
        mismatch_counts: list[int] = []
        max_abs_errors: list[int] = []

        for repetition in range(args.repetitions):
            latency_ms, registers = run_once(
                dma=dma,
                mmio=mmio,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                input_frames=input_reference.shape[0],
                output_words=golden_logits.size,
                session_id=args.session_id,
                timeout_s=args.timeout,
            )

            actual = np.asarray(output_buffer, dtype=np.int16).copy()
            difference = actual.astype(np.int32) - golden_logits.astype(np.int32)
            mismatch_count = int(np.count_nonzero(difference))
            max_abs_error = int(np.max(np.abs(difference))) if difference.size else 0

            latencies_ms.append(latency_ms)
            run_registers.append(registers)
            mismatch_counts.append(mismatch_count)
            max_abs_errors.append(max_abs_error)
            print(
                f"run={repetition + 1}/{args.repetitions} "
                f"latency_ms={latency_ms:.6f} mismatches={mismatch_count} "
                f"max_abs_error={max_abs_error} status=0x{registers['status']:08x} "
                f"errors=0x{registers['error_flags']:08x}",
                flush=True,
            )

        final_registers = run_registers[-1]
        pl_transaction_ms = final_registers["total_cycle_count"] / (PL_CLOCK_HZ / 1000.0)
        max_frame_ms = final_registers["max_frame_cycles"] / (PL_CLOCK_HZ / 1000.0)

        checks = {
            "bit_exact_all_runs": all(value == 0 for value in mismatch_counts),
            "status_done_no_error": all(
                (registers["status"] & 0xF) == 0x6 for registers in run_registers
            ),
            "error_flags_zero": all(
                registers["error_flags"] == 0 for registers in run_registers
            ),
            "input_bytes_match": final_registers["input_byte_count"]
            == input_reference.size,
            "input_frames_match": final_registers["input_frame_count"]
            == input_reference.shape[0],
            "output_words_match": final_registers["output_word_count"]
            == golden_logits.size,
            # 核内 result_frame_done 表示卷积预热后的有效处理帧；时间抽取后
            # 真正写出的 logit 帧数由 output_word_count / 32 单独校验。
            "post_warmup_frames_match": final_registers["result_frame_count"]
            == input_reference.shape[0],
            "flush_frames_match": final_registers["flush_completed_count"] == 6,
        }

        result = {
            "schema_version": 1,
            "test": "PYNQ-Z2 DDR-DMA-H144-DMA-DDR bit-exact test",
            "platform": {
                "hostname": platform.node(),
                "kernel": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "bitstream": bitstream_path.name,
                "accelerator_base_address": accelerator_base,
                "accelerator_address_range": accelerator_range,
                "pl_clock_hz": PL_CLOCK_HZ,
            },
            "trace": {
                "file": trace_path.name,
                "session_id": args.session_id,
                "input_shape": list(input_reference.shape),
                "input_dtype": str(input_reference.dtype),
                "input_bytes": int(input_reference.nbytes),
                "golden_logits_shape": list(golden_logits.shape),
                "golden_logits_dtype": str(golden_logits.dtype),
                "output_bytes": int(golden_logits.nbytes),
            },
            "repetitions": args.repetitions,
            "latency_ms": {
                "all": latencies_ms,
                "mean": float(statistics.fmean(latencies_ms)),
                "median": float(statistics.median(latencies_ms)),
                "p95": percentile(latencies_ms, 95),
                "p99": percentile(latencies_ms, 99),
                "minimum": min(latencies_ms),
                "maximum": max(latencies_ms),
            },
            "hardware_counters": {
                "pl_transaction_ms": pl_transaction_ms,
                "max_frame_ms": max_frame_ms,
                "final": final_registers,
                "all_runs": run_registers,
            },
            "comparison": {
                "mismatch_counts": mismatch_counts,
                "max_abs_errors": max_abs_errors,
                "total_output_words_per_run": int(golden_logits.size),
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["passed"] else 2
    finally:
        input_buffer.close()
        output_buffer.close()


if __name__ == "__main__":
    raise SystemExit(main())
