"""在官方 228 条 HeldOutBlocks 上评测最终 H144 与稠密基线。

本脚本只使用开发阶段冻结的解码参数，不在官方测试集上搜索阈值：

    start_threshold = 0.3
    start_delay = 15

输出包含逐句预测、总体 CER/WER、逐会话指标、数据划分指纹与 checkpoint
SHA-256，便于论文复核。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader

from brain_to_text_main_v5 import compute_error_rates, edit_distance
from paper_v5_pipeline import (
    PaperBrainToTextV5,
    PaperV5EvalDataset,
    SentenceLabelRecord,
    collate_variable_paper_batch,
    decode_paper_v5_outputs,
    render_transcript,
)
from recurrent_fixed_point_model import RecurrentFixedPointH144
from train_compact_bci import load_frozen_label_cache


DEFAULT_THRESHOLD = 0.3
DEFAULT_DELAY = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_fingerprint(records: Sequence[SentenceLabelRecord]) -> str:
    rows = [
        [record.session_name, record.sentence_index, record.block_num, record.transcript]
        for record in records
    ]
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def error_counts(references: Sequence[str], hypotheses: Sequence[str]) -> Dict[str, int | float]:
    """同时保留分子和分母，避免只报告四舍五入后的 CER/WER。"""

    char_errors = 0
    char_total = 0
    word_errors = 0
    word_total = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_display = render_transcript(reference)
        hypothesis_display = render_transcript(hypothesis)
        char_errors += edit_distance(list(reference_display), list(hypothesis_display))
        char_total += max(1, len(reference_display))
        reference_words = reference_display.split()
        hypothesis_words = hypothesis_display.split()
        word_errors += edit_distance(reference_words, hypothesis_words)
        word_total += max(1, len(reference_words))
    return {
        "char_errors": char_errors,
        "char_total": char_total,
        "word_errors": word_errors,
        "word_total": word_total,
        "cer": char_errors / max(1, char_total),
        "wer": word_errors / max(1, word_total),
    }


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_final_model(checkpoint_path: Path, device: torch.device) -> tuple[RecurrentFixedPointH144, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RecurrentFixedPointH144(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    specification = checkpoint["fixed_point_specification"]
    model.activation_mode = str(specification.get("activation_mode", "lut"))
    model.lut_entries = int(specification.get("lut_entries") or 512)
    model.lut_range = float(specification.get("lut_range", 8.0))
    model.quantization_enabled = True
    model.set_observer_mode(False, False)
    model.to(device).eval()
    return model, checkpoint


def load_final_model_fp32(
    checkpoint_path: Path, device: torch.device
) -> tuple[RecurrentFixedPointH144, dict]:
    """读取同一 QAT 权重，但旁路全部量化器，用于拆分架构误差和定点误差。"""

    model, checkpoint = load_final_model(checkpoint_path, device)
    model.quantization_enabled = False
    model.activation_mode = "exact"
    for quantizer in model.quantizers().values():
        quantizer.bypass = True
    model.eval()
    return model, checkpoint


def load_dense_model(checkpoint_path: Path, device: torch.device) -> tuple[PaperBrainToTextV5, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PaperBrainToTextV5(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(
    name: str,
    model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    output_dir: Path,
    threshold: float,
    delay: int,
) -> Dict[str, object]:
    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    batch_count = 0
    for batch in loader:
        batch_count += 1
        inputs = batch["inputs"].to(device)
        lengths = batch["input_lengths"].to(device)
        session_ids = batch["session_ids"].to(device)
        logits, _ = model(inputs, lengths, session_ids)
        logits = logits.float().cpu()
        lengths_cpu = lengths.cpu().tolist()

        for row_index, length in enumerate(lengths_cpu):
            reference = str(batch["transcripts"][row_index])
            hypothesis = decode_paper_v5_outputs(
                logits[row_index, : int(length)],
                int(length),
                start_threshold=threshold,
                start_delay=delay,
            )
            rows.append(
                {
                    "session": str(batch["session_names"][row_index]),
                    "sentence_index": int(batch["sentence_indices"][row_index].item()),
                    "block_num": int(batch["block_nums"][row_index].item()),
                    "length": int(length),
                    "reference": reference,
                    "hypothesis": hypothesis,
                }
            )

        # 每批落盘一次，长评测中断后仍可核查已经完成的预测。
        write_json(output_dir / f"{name}_predictions.partial.json", rows)
        elapsed = time.perf_counter() - started
        print(
            f"[{name}] batch={batch_count} records={len(rows)} elapsed={elapsed:.1f}s",
            flush=True,
        )

    references = [str(row["reference"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    overall = error_counts(references, hypotheses)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session"])].append(row)
    by_session = {
        session: {
            "records": len(session_rows),
            **error_counts(
                [str(row["reference"]) for row in session_rows],
                [str(row["hypothesis"]) for row in session_rows],
            ),
        }
        for session, session_rows in sorted(grouped.items())
    }
    elapsed = time.perf_counter() - started
    result = {
        "name": name,
        "records": len(rows),
        "start_threshold": threshold,
        "start_delay": delay,
        "elapsed_seconds": elapsed,
        "overall": overall,
        "by_session": by_session,
        "predictions": rows,
    }
    write_json(output_dir / f"{name}_results.json", result)
    (output_dir / f"{name}_predictions.partial.json").unlink(missing_ok=True)
    print(
        f"[{name}] COMPLETE CER={100*float(overall['cer']):.4f}% "
        f"WER={100*float(overall['wer']):.4f}% elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=root / "Datasets")
    parser.add_argument("--label_cache", type=Path, default=root / "paper_v5_label_cache.pt")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "artifacts/h144_recurrent_fixed_v1_final/experiment/fpga_candidate.pt",
    )
    parser.add_argument(
        "--baseline_checkpoint",
        type=Path,
        default=root / "brain_to_text_v5_paper_best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root / "artifacts/h144_official_heldout_final",
    )
    parser.add_argument(
        "--mode",
        choices=("final", "final_fp32", "baseline", "both", "all"),
        default="both",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    payload = load_frozen_label_cache(str(args.label_cache), str(args.data_root))
    heldout = [record for record in payload["records"] if record.split == "val"]
    if len(heldout) != 228:
        raise RuntimeError(f"Expected 228 official HeldOutBlocks records, got {len(heldout)}")
    if args.max_records > 0:
        heldout = heldout[: args.max_records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PaperV5EvalDataset(
        str(payload["datasets_root"]),
        heldout,
        bin_size=2,
        training_steps_root=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_paper_batch,
    )

    protocol = {
        "cv_part": "HeldOutBlocks",
        "records": len(heldout),
        "full_official_record_count": 228,
        "fixed_start_threshold": DEFAULT_THRESHOLD,
        "fixed_start_delay": DEFAULT_DELAY,
        "decode_grid_search": False,
        "split_fingerprint_sha256": split_fingerprint(heldout),
        "label_cache_sha256": sha256_file(args.label_cache),
        "data_root": str(args.data_root.resolve()),
    }
    write_json(args.output_dir / "protocol.json", protocol)

    results: Dict[str, object] = {"protocol": protocol, "models": {}}
    if args.mode in ("final", "both", "all"):
        final_model, final_checkpoint = load_final_model(args.checkpoint, device)
        result = evaluate(
            "h144_w8a8_h16_lut512",
            final_model,
            loader,
            device,
            args.output_dir,
            DEFAULT_THRESHOLD,
            DEFAULT_DELAY,
        )
        results["models"]["h144_w8a8_h16_lut512"] = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_stage": final_checkpoint.get("stage"),
            "parameters": int(final_model.hardware_profile()["parameters"]),
            "mac_per_20ms": float(final_model.hardware_profile()["mac_per_20ms_step"]),
            **{key: value for key, value in result.items() if key != "predictions"},
        }
        write_json(args.output_dir / "summary.json", results)

    if args.mode in ("final_fp32", "all"):
        fp32_model, fp32_checkpoint = load_final_model_fp32(args.checkpoint, device)
        result = evaluate(
            "h144_qat_weights_fp32_execution",
            fp32_model,
            loader,
            device,
            args.output_dir,
            DEFAULT_THRESHOLD,
            DEFAULT_DELAY,
        )
        results["models"]["h144_qat_weights_fp32_execution"] = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_stage": fp32_checkpoint.get("stage"),
            "purpose": "diagnostic_only_quantization_ablation",
            "parameters": int(fp32_model.hardware_profile()["parameters"]),
            "mac_per_20ms": float(fp32_model.hardware_profile()["mac_per_20ms_step"]),
            **{key: value for key, value in result.items() if key != "predictions"},
        }
        write_json(args.output_dir / "summary.json", results)

    if args.mode in ("baseline", "both", "all"):
        dense_model, dense_checkpoint = load_dense_model(args.baseline_checkpoint, device)
        result = evaluate(
            "dense_512_gru",
            dense_model,
            loader,
            device,
            args.output_dir,
            DEFAULT_THRESHOLD,
            DEFAULT_DELAY,
        )
        results["models"]["dense_512_gru"] = {
            "checkpoint": str(args.baseline_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.baseline_checkpoint),
            "checkpoint_epoch": dense_checkpoint.get("epoch"),
            "parameters": 3_048_352,
            "mac_per_20ms": 1_438_553.6,
            **{key: value for key, value in result.items() if key != "predictions"},
        }
        write_json(args.output_dir / "summary.json", results)

    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
