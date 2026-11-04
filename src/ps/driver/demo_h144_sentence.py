#!/usr/bin/env python3
"""竞赛演示：选择一条官方句子，在 PYNQ-Z2 上解码并显示文字。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pynq import MMIO, Overlay, allocate

from run_h144_dma_test import run_once


VOCAB = list("abcdefghijklmnopqrstuvwxyz>,'~?")


def decode_logits(logits_down: np.ndarray, length: int) -> str:
    """复现固定 threshold=0.3、delay=15 的事件解码器。"""
    logits = np.repeat(logits_down.astype(np.float64) / 256.0, repeats=5, axis=0)
    logits = logits[:length]
    character_index = np.argmax(logits[:, :-1], axis=1)
    onset_logits = np.clip(logits[:, -1], -40.0, 40.0)
    onset_probability = 1.0 / (1.0 + np.exp(-onset_logits))
    crossings = np.flatnonzero(
        (onset_probability[:-1] < 0.3) & (onset_probability[1:] > 0.3)
    )
    decoded = []
    for crossing in crossings:
        character_time = min(int(crossing) + 15, length - 1)
        decoded.append(VOCAB[int(character_index[character_time])])
    return "".join(decoded)


def display_text(text: str) -> str:
    return " ".join(text.replace(">", " ").replace("~", ".").split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

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

    mismatch_count = int(np.count_nonzero(actual != golden_logits))
    hypothesis = decode_logits(actual, input_reference.shape[0])
    reference = str(metadata["reference"])
    result = {
        "record_index": args.record,
        "session": metadata["session"],
        "sentence_index": metadata["sentence_index"],
        "input_frames": int(input_reference.shape[0]),
        "signal_duration_seconds": input_reference.shape[0] * 0.020,
        "board_latency_seconds": latency_ms / 1000.0,
        "real_time_factor": latency_ms / (input_reference.shape[0] * 20.0),
        "reference": display_text(reference),
        "hypothesis": display_text(hypothesis),
        "bit_mismatches": mismatch_count,
        "error_flags": int(registers["error_flags"]),
        "passed": mismatch_count == 0 and int(registers["error_flags"]) == 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
