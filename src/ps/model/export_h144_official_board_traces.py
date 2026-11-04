"""导出官方 228 条 HeldOutBlocks 的 FPGA 输入及 bit-exact 整数输出。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bit_exact_h144_golden import IntegerH144Golden
from paper_v5_pipeline import PaperV5EvalDataset, collate_variable_paper_batch
from recurrent_fixed_point_model import RecurrentFixedPointH144
from train_compact_bci import load_frozen_label_cache


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "Datasets")
    parser.add_argument("--label-cache", type=Path, default=root / "paper_v5_label_cache.pt")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "artifacts/h144_recurrent_fixed_v1_final/experiment/fpga_candidate.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "artifacts/h144_recurrent_fixed_v1_final/board/pynq_z2/official_traces",
    )
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RecurrentFixedPointH144(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    specification = checkpoint["fixed_point_specification"]
    if str(specification.get("activation_mode", "lut")) != "lut":
        raise RuntimeError("The FPGA golden runtime requires LUT nonlinearities")
    model.activation_mode = "lut"
    model.lut_entries = int(specification.get("lut_entries") or 512)
    model.lut_range = float(specification.get("lut_range", 8.0))
    model.quantization_enabled = True
    model.set_observer_mode(False, False)
    model.eval()
    golden = IntegerH144Golden(model)

    payload = load_frozen_label_cache(str(args.label_cache), str(args.data_root))
    heldout = [record for record in payload["records"] if record.split == "val"]
    if len(heldout) != 228:
        raise RuntimeError(f"Expected 228 official records, got {len(heldout)}")
    if args.max_records > 0:
        heldout = heldout[: args.max_records]

    dataset = PaperV5EvalDataset(
        str(payload["datasets_root"]), heldout, bin_size=2, training_steps_root=None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_paper_batch,
    )

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for record_index, batch in enumerate(loader):
        length = int(batch["input_lengths"][0].item())
        session_id = int(batch["session_ids"][0].item())
        inputs = batch["inputs"][0, :length]

        integer_logits, trace = golden.forward_integer(inputs, session_id)
        input_integer = trace["input"].numpy().astype(np.int8, copy=False)
        logits_down = trace["logits_down"].numpy().astype(np.int16, copy=False)
        overflow_count = sum(
            int(value) for value in trace["accumulator_overflow_count"].values()
        )
        if overflow_count != 0:
            raise RuntimeError(
                f"Integer accumulator overflow in official record {record_index}"
            )

        trace_name = f"official_{record_index:03d}.npz"
        trace_path = args.output_dir / trace_name
        np.savez_compressed(trace_path, input=input_integer, logits_down=logits_down)
        records.append(
            {
                "record_index": record_index,
                "trace_file": trace_name,
                "trace_sha256": sha256_file(trace_path),
                "session": str(batch["session_names"][0]),
                "session_id": session_id,
                "sentence_index": int(batch["sentence_indices"][0].item()),
                "block_num": int(batch["block_nums"][0].item()),
                "reference": str(batch["transcripts"][0]),
                "input_frames": int(input_integer.shape[0]),
                "input_bytes": int(input_integer.nbytes),
                "output_frames": int(logits_down.shape[0]),
                "output_words": int(logits_down.size),
                "output_bytes": int(logits_down.nbytes),
                "accumulator_overflow_count": overflow_count,
            }
        )

        manifest = {
            "schema_version": 1,
            "protocol": "official HeldOutBlocks board bit-exact export",
            "official_record_count": 228,
            "exported_record_count": len(records),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "label_cache_sha256": sha256_file(args.label_cache),
            "input_bin_ms": 20,
            "elapsed_seconds": time.perf_counter() - started,
            "complete": len(records) == len(heldout),
            "records": records,
        }
        write_json_atomic(args.output_dir / "manifest.json", manifest)
        print(
            f"export={record_index + 1}/{len(heldout)} session={records[-1]['session']} "
            f"frames={length} elapsed_s={manifest['elapsed_seconds']:.1f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
