from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from compact_bci_model import CompactBrainToTextGRU
from brain_to_text_main_v5 import compute_error_rates
from paper_v5_pipeline import (
    PaperBrainToTextV5,
    PaperV5EvalDataset,
    PaperV5RealDataset,
    PaperV5SyntheticDataset,
    SentenceLabelRecord,
    build_snippet_library,
    collate_variable_paper_batch,
    collect_word_list,
    compute_paper_v5_loss,
    decode_paper_v5_outputs,
    evaluate_paper_v5,
)
from spatiotemporal_discriminability_atlas import split_train_internal


BASELINE_PARAMETERS = 3_048_352
BASELINE_MAC_PER_20MS = 1_438_553.6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frozen_label_cache(cache_path: str, data_root: str | None) -> Dict[str, object]:
    """直接读取既有官方标签缓存，避免远端缺少 Step2 目录时意外重建标签。"""

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    payload["records"] = [
        item if isinstance(item, SentenceLabelRecord) else SentenceLabelRecord.from_serializable(item)
        for item in payload["records"]
    ]
    resolved_root = os.path.abspath(data_root or str(payload["datasets_root"]))
    if not os.path.isdir(resolved_root):
        raise FileNotFoundError(f"Dataset root does not exist: {resolved_root}")
    payload["datasets_root"] = resolved_root
    # 训练和验证目标都由缓存中的官方 letter_starts 构造；不访问已经不存在的旧绝对路径。
    payload["training_steps_root"] = None
    return payload


def split_fingerprint(records: Iterable[SentenceLabelRecord]) -> str:
    rows = [
        [record.session_name, record.sentence_index, record.block_num, record.split, record.transcript]
        for record in records
    ]
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        **batch,
        "inputs": batch["inputs"].to(device, non_blocking=True),
        "char_prob_target": batch["char_prob_target"].to(device, non_blocking=True),
        "char_start_target": batch["char_start_target"].to(device, non_blocking=True),
        "err_weights": batch["err_weights"].to(device, non_blocking=True),
        "input_lengths": batch["input_lengths"].to(device, non_blocking=True),
        "session_ids": batch["session_ids"].to(device, non_blocking=True),
    }


def collate_training_batch(batch: list[Dict[str, object]]) -> Dict[str, object]:
    """训练只保留共同字段，兼容没有 session_name 等元数据的合成样本。"""

    return {
        "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
        "char_prob_target": torch.stack([item["char_prob_target"] for item in batch], dim=0),
        "char_start_target": torch.stack([item["char_start_target"] for item in batch], dim=0),
        "err_weights": torch.stack([item["err_weights"] for item in batch], dim=0),
        "input_lengths": torch.tensor([int(item["input_length"]) for item in batch], dtype=torch.long),
        "session_ids": torch.tensor([int(item["session_id"]) for item in batch], dtype=torch.long),
        "transcripts": [str(item.get("transcript", "")) for item in batch],
        "display_texts": [str(item.get("display_text", "")) for item in batch],
    }


def merge_batches(*batches: Dict[str, object] | None) -> Dict[str, object] | None:
    valid = [batch for batch in batches if batch is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return {
        "inputs": torch.cat([batch["inputs"] for batch in valid], dim=0),
        "char_prob_target": torch.cat([batch["char_prob_target"] for batch in valid], dim=0),
        "char_start_target": torch.cat([batch["char_start_target"] for batch in valid], dim=0),
        "err_weights": torch.cat([batch["err_weights"] for batch in valid], dim=0),
        "input_lengths": torch.cat([batch["input_lengths"] for batch in valid], dim=0),
        "session_ids": torch.cat([batch["session_ids"] for batch in valid], dim=0),
        "transcripts": sum((list(batch["transcripts"]) for batch in valid), []),
        "display_texts": sum((list(batch["display_texts"]) for batch in valid), []),
    }


def add_input_noise(
    inputs: torch.Tensor,
    white_noise_sd: float,
    constant_offset_sd: float,
    random_walk_sd: float,
) -> torch.Tensor:
    if white_noise_sd > 0:
        inputs = inputs + torch.randn_like(inputs) * white_noise_sd
    if constant_offset_sd > 0:
        inputs = inputs + torch.randn(
            (inputs.shape[0], 1, inputs.shape[2]), device=inputs.device, dtype=inputs.dtype
        ) * constant_offset_sd
    if random_walk_sd > 0:
        walk = torch.randn_like(inputs) * random_walk_sd
        inputs = inputs + torch.cumsum(walk, dim=1)
    return inputs


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    err_weights: torch.Tensor,
    output_delay: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if output_delay > 0 and student_logits.shape[1] > output_delay:
        student_logits = student_logits[:, output_delay:, :]
        teacher_logits = teacher_logits[:, output_delay:, :]
        err_weights = err_weights[:, :-output_delay]

    valid = err_weights.sum().clamp_min(1.0)
    student_char = student_logits[:, :, :-1] / temperature
    teacher_char = teacher_logits[:, :, :-1] / temperature
    teacher_prob = F.softmax(teacher_char, dim=-1)
    char_kl = F.kl_div(F.log_softmax(student_char, dim=-1), teacher_prob, reduction="none").sum(dim=-1)
    char_kl = (char_kl * err_weights).sum() / valid * (temperature**2)

    student_start = torch.sigmoid(student_logits[:, :, -1])
    teacher_start = torch.sigmoid(teacher_logits[:, :, -1])
    start_mse = (((student_start - teacher_start) ** 2) * err_weights).sum() / valid
    # 两项分开返回，便于 H144-v2 显式满足 character KD > onset KD。
    return char_kl, start_mse, {
        "distill_char_kl": float(char_kl.item()),
        "distill_start_mse": float(start_mse.item()),
    }


def parse_number_grid(value: str, cast) -> List[float | int]:
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("解码网格不能为空。")
    return values


def evaluate_paper_v5_grid(
    model: CompactBrainToTextGRU,
    data_loader: Iterable[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """只执行一次网络前向，在403/99开发集上选择 WER 最优解码门限。"""

    thresholds = [float(value) for value in parse_number_grid(args.decode_thresholds, float)]
    delays = [int(value) for value in parse_number_grid(args.decode_delays, int)]
    logits_list: List[torch.Tensor] = []
    lengths: List[int] = []
    references: List[str] = []
    total_loss = 0.0
    batches = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            moved = move_batch(batch, device)
            with autocast_context(device, args.amp_enabled):
                logits, _ = model(
                    moved["inputs"], moved["input_lengths"], moved["session_ids"]
                )
                loss, _ = compute_paper_v5_loss(
                    logits=logits,
                    char_prob_target=moved["char_prob_target"],
                    char_start_target=moved["char_start_target"],
                    err_weights=moved["err_weights"],
                    output_delay=args.output_delay,
                )
            total_loss += float(loss.item())
            batches += 1
            cpu_logits = logits.float().cpu()
            for row, length in enumerate(moved["input_lengths"].cpu().tolist()):
                logits_list.append(cpu_logits[row, : int(length)].clone())
                lengths.append(int(length))
            references.extend(str(value) for value in batch["transcripts"])

    candidates: List[Dict[str, object]] = []
    for threshold in thresholds:
        for delay in delays:
            hypotheses = [
                decode_paper_v5_outputs(logits, length, threshold, delay)
                for logits, length in zip(logits_list, lengths)
            ]
            errors = compute_error_rates(references, hypotheses)
            candidates.append(
                {
                    "cer": float(errors["cer"]),
                    "wer": float(errors["wer"]),
                    "start_threshold": threshold,
                    "start_delay": delay,
                }
            )
    best = min(candidates, key=lambda item: (float(item["wer"]), float(item["cer"])))
    best["loss"] = total_loss / max(batches, 1)
    best["decode_candidates"] = len(candidates)
    return best


def l2_penalty(model: nn.Module) -> torch.Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2 and "bias" not in name and "start_state" not in name:
            penalty = penalty + torch.sum(parameter.float() ** 2)
    return penalty


def train_one_epoch(
    model: CompactBrainToTextGRU,
    teacher: PaperBrainToTextV5 | None,
    optimizer: optim.Optimizer,
    real_loader: DataLoader,
    synth_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    totals: Dict[str, float] = defaultdict(float)
    steps = 0
    start_time = time.time()
    real_iter = iter(real_loader)
    synth_iter = iter(synth_loader)
    max_steps = max(len(real_loader), len(synth_loader))
    distill_active = bool(
        teacher is not None
        and (args.distill_char_weight > 0 or args.distill_onset_weight > 0)
        and (
            epoch <= args.distill_warmup_epochs
            or (epoch - args.distill_warmup_epochs) % args.distill_every == 0
        )
    )

    for step_index in range(max_steps):
        try:
            real_batch = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader)
            real_batch = next(real_iter)
        try:
            synth_batch = next(synth_iter)
        except StopIteration:
            synth_iter = iter(synth_loader)
            synth_batch = next(synth_iter)

        batch = merge_batches(synth_batch, real_batch)
        if batch is None:
            continue
        batch = move_batch(batch, device)
        batch["inputs"] = add_input_noise(
            batch["inputs"], args.white_noise_sd, args.constant_offset_sd, args.random_walk_sd
        )
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, args.amp_enabled):
            teacher_logits = None
            if distill_active:
                assert teacher is not None
                with torch.no_grad():
                    teacher_logits, _ = teacher(
                        batch["inputs"], batch["input_lengths"], batch["session_ids"]
                    )
            student_logits, _ = model(batch["inputs"], batch["input_lengths"], batch["session_ids"])
            supervised, supervised_stats = compute_paper_v5_loss(
                logits=student_logits,
                char_prob_target=batch["char_prob_target"],
                char_start_target=batch["char_start_target"],
                err_weights=batch["err_weights"],
                output_delay=args.output_delay,
            )
            if distill_active:
                assert teacher_logits is not None
                char_kd, onset_kd, distill_stats = compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    err_weights=batch["err_weights"],
                    output_delay=args.output_delay,
                    temperature=args.distill_temperature,
                )
            else:
                char_kd = supervised.new_zeros(())
                onset_kd = supervised.new_zeros(())
                distill_stats = {"distill_char_kl": 0.0, "distill_start_mse": 0.0}
            distill = (
                float(args.distill_char_weight) * char_kd
                + float(args.distill_onset_weight) * onset_kd
            )
            penalty = l2_penalty(model)
            loss = supervised + distill + args.l2_scale * penalty

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "supervised": float(supervised.item()),
            "distill": float(distill.item()),
            "distill_active": float(distill_active),
            "l2": float(penalty.item()),
            **supervised_stats,
            **distill_stats,
        }
        for key, value in stats.items():
            totals[key] += value
        steps += 1
        if args.train_log_every > 0 and steps % args.train_log_every == 0:
            print(
                f"           train {steps}/{max_steps} | loss={totals['loss']/steps:.4f} | "
                f"elapsed={(time.time()-start_time)/60:.1f} min",
                flush=True,
            )
        max_train_batches = int(getattr(args, "max_train_batches", 0))
        if max_train_batches > 0 and step_index + 1 >= max_train_batches:
            break

    return {key: value / max(1, steps) for key, value in totals.items()}


def save_checkpoint(
    path: str,
    model: CompactBrainToTextGRU,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    best_cer: float,
    best_wer: float,
    best_selection_score: float,
    split_hash: str,
    metrics: Dict[str, object] | None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_cer": best_cer,
            "best_wer": best_wer,
            "best_selection_metric": args.selection_metric,
            "best_selection_score": best_selection_score,
            "model_config": model.export_config(),
            "hardware_profile": model.hardware_profile(),
            "split_fingerprint": split_hash,
            "validation_metrics": metrics,
            "train_config": vars(args),
        },
        path,
    )


def append_metrics(path: str, row: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact FPGA-oriented BCI GRU with teacher distillation.")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--label_cache", default="paper_v5_label_cache.pt")
    parser.add_argument("--teacher_checkpoint", default="brain_to_text_v5_paper_best.pt")
    parser.add_argument("--save_path", default="compact_bci_best.pt")
    parser.add_argument("--save_last_path", default="compact_bci_last.pt")
    parser.add_argument("--metrics_path", default="compact_bci_metrics.jsonl")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--reset_optimizer_on_resume",
        action="store_true",
        help="恢复模型权重和最佳 CER，但用命令行学习率重新建立优化器与调度器。",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size_real", type=int, default=40)
    parser.add_argument("--batch_size_synth", type=int, default=24)
    parser.add_argument("--synthetic_samples_per_epoch", type=int, default=480)
    parser.add_argument("--time_steps", type=int, default=1200)
    parser.add_argument("--bin_size", type=int, default=2)
    parser.add_argument("--output_delay", type=int, default=50)
    parser.add_argument("--bottleneck_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--adapter_rank", type=int, default=16)
    parser.add_argument("--temporal_kernel", type=int, default=13)
    parser.add_argument("--causal_conv", action="store_true")
    parser.add_argument("--skip_len", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_end", type=float, default=1e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-1)
    parser.add_argument("--l2_scale", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--distill_weight", type=float, default=0.5)
    parser.add_argument(
        "--distill_char_weight",
        type=float,
        default=None,
        help="字符后验KD权重；未指定时兼容旧版 distill_weight。",
    )
    parser.add_argument(
        "--distill_onset_weight",
        type=float,
        default=None,
        help="onset KD权重；未指定时兼容旧版 5*distill_weight。",
    )
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--distill_warmup_epochs", type=int, default=0)
    parser.add_argument("--distill_every", type=int, default=1)
    parser.add_argument("--white_noise_sd", type=float, default=1.2)
    parser.add_argument("--constant_offset_sd", type=float, default=0.6)
    parser.add_argument("--random_walk_sd", type=float, default=0.02)
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["on", "off"], default="on")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--validate_every", type=int, default=5)
    parser.add_argument("--train_log_every", type=int, default=5)
    parser.add_argument("--val_log_every", type=int, default=5)
    parser.add_argument("--start_threshold", type=float, default=0.3)
    parser.add_argument("--start_delay", type=int, default=15)
    parser.add_argument(
        "--validation_protocol",
        choices=("official_heldout", "internal_last_block"),
        default="internal_last_block",
    )
    parser.add_argument(
        "--selection_metric", choices=("wer", "cer"), default="wer"
    )
    parser.add_argument("--decode_grid", action="store_true")
    parser.add_argument(
        "--decode_thresholds", default="0.20,0.25,0.30,0.35,0.40,0.50"
    )
    parser.add_argument("--decode_delays", default="10,15,20,25")
    parser.add_argument(
        "--patience_validations",
        type=int,
        default=0,
        help="连续多少次验证不改善后早停；0表示禁用。",
    )
    parser.add_argument("--experiment_stage", default="legacy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.distill_every < 1:
        raise ValueError("--distill_every must be at least 1")
    if args.distill_char_weight is None:
        args.distill_char_weight = float(args.distill_weight)
    if args.distill_onset_weight is None:
        args.distill_onset_weight = 5.0 * float(args.distill_weight)
    if args.distill_char_weight < 0 or args.distill_onset_weight < 0:
        raise ValueError("蒸馏权重不能为负数。")
    if args.patience_validations < 0:
        raise ValueError("patience_validations 不能为负数。")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.amp_enabled = args.amp == "on" and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    payload = load_frozen_label_cache(args.label_cache, args.data_root)
    records = payload["records"]
    official_train_records = [record for record in records if record.split == "train"]
    official_heldout_records = [record for record in records if record.split == "val"]
    internal_blocks: Dict[str, int] = {}
    if args.validation_protocol == "internal_last_block":
        train_records, val_records, internal_blocks = split_train_internal(
            official_train_records
        )
        heldout_signals_loaded = False
    else:
        train_records = official_train_records
        val_records = official_heldout_records
        heldout_signals_loaded = True
    fingerprint = split_fingerprint([*train_records, *val_records])
    print(
        f"Frozen split | protocol={args.validation_protocol} | "
        f"train={len(train_records)} | val={len(val_records)} | "
        f"official_heldout_signals_loaded={heldout_signals_loaded} | "
        f"sessions={len(payload['session_names'])} | fingerprint={fingerprint}",
        flush=True,
    )
    if internal_blocks:
        print(
            "Internal development blocks | "
            + json.dumps(internal_blocks, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = PaperBrainToTextV5(**teacher_checkpoint["model_config"])
    teacher.load_state_dict(teacher_checkpoint["model_state"])
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    model = CompactBrainToTextGRU(
        input_size=192,
        bottleneck_size=args.bottleneck_size,
        hidden_size=args.hidden_size,
        num_sessions=len(payload["session_names"]),
        adapter_rank=args.adapter_rank,
        temporal_kernel=args.temporal_kernel,
        causal_conv=args.causal_conv,
        skip_len=args.skip_len,
        dropout=args.dropout,
    ).to(device)
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model_state"])
    elif not args.random_init:
        model.initialize_from_teacher(teacher)

    distillation_requested = (
        args.distill_char_weight > 0 or args.distill_onset_weight > 0
    )
    if not distillation_requested:
        # B0只复用相同SVD初始化，训练阶段不保留Dense512，避免无效显存占用。
        del teacher
        teacher = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    profile = model.hardware_profile()
    profile["parameter_compression"] = BASELINE_PARAMETERS / profile["parameters"]
    profile["mac_reduction"] = BASELINE_MAC_PER_20MS / profile["mac_per_20ms_step"]
    print("Hardware profile | " + json.dumps(profile, ensure_ascii=False, sort_keys=True), flush=True)
    if args.profile_only:
        return

    print("Building snippet library from the frozen training split...", flush=True)
    training_payload = {**payload, "records": list(train_records)}
    snippet_library = build_snippet_library(
        payload["datasets_root"], training_payload, training_steps_root=None
    )
    words, rare_word_indices = collect_word_list(train_records)
    real_dataset = PaperV5RealDataset(
        payload["datasets_root"], train_records, args.time_steps, args.bin_size, training_steps_root=None
    )
    synthetic_dataset = PaperV5SyntheticDataset(
        snippet_library,
        words,
        rare_word_indices,
        args.time_steps,
        args.bin_size,
        args.synthetic_samples_per_epoch,
    )
    val_dataset = PaperV5EvalDataset(
        payload["datasets_root"], val_records, args.bin_size, training_steps_root=None
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    real_loader = DataLoader(
        real_dataset,
        batch_size=args.batch_size_real,
        shuffle=True,
        collate_fn=collate_training_batch,
        **loader_options,
    )
    synth_loader = DataLoader(
        synthetic_dataset,
        batch_size=args.batch_size_synth,
        shuffle=True,
        collate_fn=collate_training_batch,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_variable_paper_batch,
        **loader_options,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    if args.epochs > 1:
        end_ratio = args.learning_rate_end / args.learning_rate
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda index: 1.0 + (end_ratio - 1.0) * index / (args.epochs - 1),
        )
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _index: 1.0)
    start_epoch = 0
    best_cer = float("inf")
    best_wer = float("inf")
    best_selection_score = float("inf")
    stale_validations = 0
    if resume_payload is not None:
        start_epoch = int(resume_payload.get("epoch", 0))
        best_cer = float(resume_payload.get("best_cer", float("inf")))
        best_wer = float(resume_payload.get("best_wer", float("inf")))
        best_selection_score = float(
            resume_payload.get(
                "best_selection_score",
                best_wer if args.selection_metric == "wer" else best_cer,
            )
        )
        if args.reset_optimizer_on_resume:
            # 精调阶段从命令行指定的学习率重新开始，并在剩余轮数内衰减。
            remaining_epochs = max(args.epochs - start_epoch, 1)
            if remaining_epochs > 1:
                end_ratio = args.learning_rate_end / args.learning_rate
                scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda index: 1.0
                    + (end_ratio - 1.0) * index / (remaining_epochs - 1),
                )
            else:
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _index: 1.0)
            print(
                f"Reset optimizer for fine-tuning | lr={args.learning_rate:.6g} | "
                f"remaining_epochs={remaining_epochs}",
                flush=True,
            )
        else:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            if "scheduler_state" in resume_payload:
                scheduler.load_state_dict(resume_payload["scheduler_state"])
        print(f"Resumed from {args.resume} at epoch {start_epoch} | best_CER={best_cer*100:.3f}%", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_enabled)

    if start_epoch == 0 and os.path.exists(args.metrics_path):
        os.remove(args.metrics_path)
    print(
        f"Training | device={device} | amp={args.amp_enabled} | hidden={args.hidden_size} | "
        f"stage={args.experiment_stage} | selection={args.selection_metric} | "
        f"KD_char={args.distill_char_weight} | KD_onset={args.distill_onset_weight}",
        flush=True,
    )
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, teacher, optimizer, real_loader, synth_loader, device, args, scaler, epoch
        )
        scheduler.step()
        print(
            f"Epoch {epoch:03d} | loss={train_metrics['loss']:.4f} | "
            f"supervised={train_metrics['supervised']:.4f} | distill={train_metrics['distill']:.4f} | "
            f"distill_active={int(train_metrics['distill_active'])} | lr={optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        validation_metrics = None
        improved = False
        should_validate = epoch == 1 or epoch == args.epochs or epoch % args.validate_every == 0
        if should_validate:
            if args.decode_grid:
                validation_metrics = evaluate_paper_v5_grid(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    args=args,
                )
            else:
                validation_metrics = evaluate_paper_v5(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    output_delay=args.output_delay,
                    amp_enabled=args.amp_enabled,
                    start_threshold=args.start_threshold,
                    start_delay=args.start_delay,
                    log_every_batches=args.val_log_every,
                )
            print(
                f"           val_loss={validation_metrics['loss']:.4f} | "
                f"CER={validation_metrics['cer']*100:.3f}% | WER={validation_metrics['wer']*100:.3f}%"
                + (
                    f" | threshold={validation_metrics['start_threshold']:.2f}"
                    f" delay={validation_metrics['start_delay']}"
                    if args.decode_grid
                    else ""
                ),
                flush=True,
            )
            current_cer = float(validation_metrics["cer"])
            current_wer = float(validation_metrics["wer"])
            current_key = (
                (current_wer, current_cer)
                if args.selection_metric == "wer"
                else (current_cer, current_wer)
            )
            best_key = (
                (best_wer, best_cer)
                if args.selection_metric == "wer"
                else (best_cer, best_wer)
            )
            if current_key < best_key:
                best_cer = current_cer
                best_wer = current_wer
                best_selection_score = current_key[0]
                stale_validations = 0
                improved = True
                save_checkpoint(
                    args.save_path,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    best_cer,
                    best_wer,
                    best_selection_score,
                    fingerprint,
                    {key: value for key, value in validation_metrics.items() if key != "examples"},
                )
                print(f"           saved best -> {args.save_path}", flush=True)
            else:
                stale_validations += 1

        row = {
            "epoch": epoch,
            "seconds": time.time() - epoch_start,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": None
            if validation_metrics is None
            else {key: value for key, value in validation_metrics.items() if key != "examples"},
        }
        append_metrics(args.metrics_path, row)
        save_checkpoint(
            args.save_last_path,
            model,
            optimizer,
            scheduler,
            args,
            epoch,
            best_cer,
            best_wer,
            best_selection_score,
            fingerprint,
            row["validation"],
        )
        if (
            should_validate
            and args.patience_validations > 0
            and stale_validations >= args.patience_validations
        ):
            print(
                f"Early stopping | stale_validations={stale_validations} | "
                f"best_{args.selection_metric}={best_selection_score:.6f}",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
