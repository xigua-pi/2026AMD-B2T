"""基于真实 Utah 阵列拓扑的 H144+CTC、空间增强和 B-first 实验。

训练只使用官方 train 记录；HeldOutBlocks 只在每个固定训练配方结束后评测一次。
四个主要消融共享同一 H144 初始权重与训练预算，避免把训练时长差异误当成结构收益。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from compact_bci_model import CompactBrainToTextGRU
from paper_v5_pipeline import (
    CHAR_DEF,
    N_CHAR_CLASSES,
    PaperV5EvalDataset,
    SentenceLabelRecord,
    collate_variable_paper_batch,
    compute_error_rates,
)


# 直接取自当前 Dryad sentences.mat 的 arrayGeometryMap。附件中的 Blackrock
# 接线表顺时针旋转 90 度后与该矩阵逐元素一致；这里以数据文件方向为准。
DRYAD_GEOMETRY = np.asarray(
    [
        [0, 79, 77, 75, 73, 71, 69, 67, 65, 0],
        [81, 80, 78, 76, 74, 72, 70, 68, 66, 2],
        [83, 84, 82, 45, 41, 39, 37, 35, 33, 1],
        [85, 86, 49, 47, 43, 40, 38, 36, 34, 3],
        [88, 87, 53, 51, 44, 42, 48, 5, 7, 4],
        [90, 89, 55, 56, 46, 50, 15, 17, 9, 6],
        [92, 91, 57, 58, 52, 54, 19, 13, 11, 8],
        [93, 94, 59, 60, 62, 21, 25, 23, 12, 10],
        [96, 63, 61, 64, 31, 29, 27, 20, 16, 14],
        [0, 95, 32, 30, 28, 26, 24, 22, 18, 0],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Variant:
    name: str
    graph_filter: bool
    dropout_mode: str
    array_mode: str


VARIANTS = {
    "ctc_baseline": Variant("ctc_baseline", False, "none", "AB"),
    "ctc_random": Variant("ctc_random", False, "random", "AB"),
    "ctc_patch": Variant("ctc_patch", False, "patch", "AB"),
    "ctc_graph_patch": Variant("ctc_graph_patch", True, "patch", "AB"),
    "ctc_b_graph_patch": Variant("ctc_b_graph_patch", True, "patch", "B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="Datasets")
    parser.add_argument("--label_cache", default="paper_v5_label_cache.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="artifacts/geometry_ctc_full")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="逗号分隔；可选：" + ",".join(VARIANTS),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--learning_rate_end", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout_probability", type=float, default=0.60)
    parser.add_argument("--patch_sizes", default="2,3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--augmentation_seed",
        type=int,
        default=None,
        help="可选；用于配对消融，使不同结构看到完全相同的通道遮挡序列。",
    )
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(label_cache: str) -> Tuple[List[SentenceLabelRecord], List[SentenceLabelRecord]]:
    payload = torch.load(label_cache, map_location="cpu", weights_only=False)
    records = [
        item if isinstance(item, SentenceLabelRecord) else SentenceLabelRecord.from_serializable(item)
        for item in payload["records"]
    ]
    return (
        [record for record in records if record.split == "train"],
        [record for record in records if record.split == "val"],
    )


def geometry_positions() -> Dict[int, Tuple[int, int]]:
    positions: Dict[int, Tuple[int, int]] = {}
    for row in range(DRYAD_GEOMETRY.shape[0]):
        for col in range(DRYAD_GEOMETRY.shape[1]):
            channel = int(DRYAD_GEOMETRY[row, col])
            if channel > 0:
                positions[channel - 1] = (row, col)
    if set(positions) != set(range(96)):
        raise RuntimeError("Utah 几何表没有恰好覆盖 local channel 1..96。")
    return positions


def build_neighbor_matrix() -> Tuple[torch.Tensor, int]:
    positions = geometry_positions()
    matrix = np.zeros((192, 192), dtype=np.float32)
    directed_edges = 0
    inverse = {position: channel for channel, position in positions.items()}
    for array_offset in (0, 96):
        for local_channel, (row, col) in positions.items():
            neighbors = []
            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = inverse.get((row + delta_row, col + delta_col))
                if neighbor is not None:
                    neighbors.append(neighbor)
            if not neighbors:
                matrix[array_offset + local_channel, array_offset + local_channel] = 1.0
            else:
                weight = 1.0 / len(neighbors)
                for neighbor in neighbors:
                    matrix[array_offset + local_channel, array_offset + neighbor] = weight
                    directed_edges += 1
    return torch.from_numpy(matrix), directed_edges


class LocalGraphFilter(nn.Module):
    """每块阵列共享一个可学习混合系数，仅增加两个标量参数。"""

    def __init__(self, enabled: bool) -> None:
        super().__init__()
        neighbor_matrix, directed_edges = build_neighbor_matrix()
        self.register_buffer("neighbor_matrix", neighbor_matrix)
        self.enabled = bool(enabled)
        self.mix = nn.Parameter(torch.zeros(2), requires_grad=self.enabled)
        self.directed_edges = int(directed_edges)
        self.degree_three_nodes = int(
            torch.count_nonzero(torch.count_nonzero(neighbor_matrix, dim=1) == 3).item()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return inputs
        neighbor_mean = torch.matmul(inputs, self.neighbor_matrix.transpose(0, 1))
        mix = self.mix.repeat_interleave(96).view(1, 1, 192)
        return inputs + mix * (neighbor_mean - inputs)

    def hardware_adds_per_step(self) -> int:
        # 邻域累加 + 与原通道做差 + 混合后回加；除以邻居数可用移位/小常数乘法。
        return self.directed_edges + 2 * 192 if self.enabled else 0

    def hardware_multiplies_per_step(self) -> int:
        # 每通道乘一次阵列共享 mix；度为 3 的节点另需乘 1/3，度 2/4 可移位。
        return 192 + self.degree_three_nodes if self.enabled else 0


class GeometryCTCModel(nn.Module):
    def __init__(self, base: CompactBrainToTextGRU, graph_filter: bool) -> None:
        super().__init__()
        self.graph = LocalGraphFilter(graph_filter)
        self.base = base

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor, session_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        filtered = self.graph(inputs)
        repeated_logits, _ = self.base(filtered, lengths, session_ids)
        logits = repeated_logits[:, :: self.base.skip_len, :]
        down_lengths = torch.div(
            lengths + self.base.skip_len - 1,
            self.base.skip_len,
            rounding_mode="floor",
        ).clamp_min(1)
        return logits, down_lengths


def initialize_ctc_model(
    checkpoint_path: str, graph_filter: bool, device: torch.device
) -> Tuple[GeometryCTCModel, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = CompactBrainToTextGRU(**checkpoint["model_config"])
    base.load_state_dict(checkpoint["model_state"])
    # 原最后一维是 char-start。CTC blank 与它语义相反，因此取反作为稳定初始化。
    with torch.no_grad():
        base.readout.weight[-1].mul_(-1.0)
        base.readout.bias[-1].mul_(-1.0)
    return GeometryCTCModel(base, graph_filter).to(device), checkpoint


def patch_channels(patch_size: int, rng: random.Random) -> List[int]:
    row0 = rng.randrange(0, 10 - patch_size + 1)
    col0 = rng.randrange(0, 10 - patch_size + 1)
    channels: List[int] = []
    for row in range(row0, row0 + patch_size):
        for col in range(col0, col0 + patch_size):
            channel = int(DRYAD_GEOMETRY[row, col])
            if channel > 0:
                channels.append(channel - 1)
    return channels


def augment_inputs(
    inputs: torch.Tensor,
    mode: str,
    probability: float,
    patch_sizes: Sequence[int],
    array_mode: str,
    rng: random.Random,
) -> torch.Tensor:
    augmented = inputs.clone()
    if array_mode == "A":
        augmented[:, :, 96:] = 0.0
    elif array_mode == "B":
        augmented[:, :, :96] = 0.0
    elif array_mode != "AB":
        raise ValueError(f"Unsupported array mode: {array_mode}")

    if mode == "none" or probability <= 0:
        return augmented

    valid_arrays = [0, 96] if array_mode == "AB" else ([0] if array_mode == "A" else [96])
    for row in range(augmented.shape[0]):
        if rng.random() >= probability:
            continue
        offset = rng.choice(valid_arrays)
        patch_size = rng.choice(list(patch_sizes))
        local_channels = patch_channels(patch_size, rng)
        if not local_channels:
            continue
        if mode == "patch":
            dropped = local_channels
        elif mode == "random":
            dropped = rng.sample(range(96), k=len(local_channels))
        else:
            raise ValueError(f"Unsupported dropout mode: {mode}")
        augmented[row, :, torch.as_tensor([offset + channel for channel in dropped])] = 0.0
    return augmented


def make_ctc_targets(transcripts: Sequence[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded: List[int] = []
    lengths: List[int] = []
    char_to_index = CHAR_DEF["strToCharIdx"]
    for transcript in transcripts:
        indices = [int(char_to_index[character]) for character in transcript]
        encoded.extend(indices)
        lengths.append(len(indices))
    return (
        torch.as_tensor(encoded, dtype=torch.long, device=device),
        torch.as_tensor(lengths, dtype=torch.long, device=device),
    )


def greedy_decode(logits: torch.Tensor, length: int) -> Tuple[str, float]:
    usable = logits[: int(length)].float()
    log_prob = F.log_softmax(usable, dim=-1)
    best = torch.argmax(log_prob, dim=-1).tolist()
    blank = N_CHAR_CLASSES
    collapsed: List[int] = []
    margins: List[float] = []
    previous = blank
    for time_index, token in enumerate(best):
        if token != blank and token != previous:
            collapsed.append(int(token))
            top_two = torch.topk(log_prob[time_index], k=2).values
            margins.append(float((top_two[0] - top_two[1]).item()))
        previous = int(token)
    text = "".join(CHAR_DEF["charListAbbr"][token] for token in collapsed)
    confidence = float(np.mean(margins)) if margins else -100.0
    return text, confidence


def train_variant(
    variant: Variant,
    args: argparse.Namespace,
    train_loader: DataLoader,
    device: torch.device,
    variant_dir: Path,
) -> GeometryCTCModel:
    model, source_checkpoint = initialize_ctc_model(args.checkpoint, variant.graph_filter, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.learning_rate_end,
    )
    ctc_loss = nn.CTCLoss(blank=N_CHAR_CLASSES, reduction="mean", zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    patch_sizes = [int(value) for value in args.patch_sizes.split(",") if value.strip()]
    variant_seed = (
        int(args.augmentation_seed)
        if args.augmentation_seed is not None
        else args.seed + sum(ord(character) for character in variant.name)
    )
    rng = random.Random(variant_seed)
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            inputs = batch["inputs"].to(device, non_blocking=True)
            lengths = batch["input_lengths"].to(device, non_blocking=True)
            session_ids = batch["session_ids"].to(device, non_blocking=True)
            inputs = augment_inputs(
                inputs,
                mode=variant.dropout_mode,
                probability=args.dropout_probability,
                patch_sizes=patch_sizes,
                array_mode=variant.array_mode,
                rng=rng,
            )
            targets, target_lengths = make_ctc_targets(batch["transcripts"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits, output_lengths = model(inputs, lengths, session_ids)
                log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
                loss = ctc_loss(log_probs, targets, output_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            total_steps += 1
            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    f"{variant.name} epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={total_loss / total_steps:.4f}",
                    flush=True,
                )
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "loss": total_loss / max(1, total_steps),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print(f"{variant.name} epoch summary: {row}", flush=True)

    variant_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "source_checkpoint": str(args.checkpoint),
            "source_model_config": source_checkpoint["model_config"],
            "variant": variant.__dict__,
            "train_args": vars(args),
            "history": history,
            "blank_index": N_CHAR_CLASSES,
        },
        variant_dir / "final.pt",
    )
    (variant_dir / "train_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model


@torch.no_grad()
def evaluate_model(
    model: GeometryCTCModel,
    loader: DataLoader,
    device: torch.device,
    array_mode: str,
) -> Dict[str, Any]:
    model.eval()
    references: List[str] = []
    hypotheses: List[str] = []
    confidences: List[float] = []
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        lengths = batch["input_lengths"].to(device, non_blocking=True)
        session_ids = batch["session_ids"].to(device, non_blocking=True)
        inputs = augment_inputs(inputs, "none", 0.0, (2,), array_mode, random.Random(0))
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits, output_lengths = model(inputs, lengths, session_ids)
        for row_index in range(inputs.shape[0]):
            hypothesis, confidence = greedy_decode(
                logits[row_index].cpu(), int(output_lengths[row_index].item())
            )
            reference = str(batch["transcripts"][row_index])
            references.append(reference)
            hypotheses.append(hypothesis)
            confidences.append(confidence)
            rows.append(
                {
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "confidence": confidence,
                }
            )
    metrics = compute_error_rates(references, hypotheses)
    return {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "mean_confidence": float(np.mean(confidences)),
        "rows": rows,
    }


def error_count(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_char in reference:
        current = [previous[0] + 1]
        for col, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col] + 1,
                    previous[col - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return int(previous[-1])


def conditional_fusion(
    train_b: Dict[str, Any],
    train_ab: Dict[str, Any],
    test_b: Dict[str, Any],
    test_ab: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """用训练集置信度分位数锁定门限，再在 held-out 集评测。"""

    train_confidence = np.asarray([row["confidence"] for row in train_b["rows"]])
    results: List[Dict[str, Any]] = []
    for fallback_target in (0.25, 0.50, 0.75):
        threshold = float(np.quantile(train_confidence, fallback_target))
        references: List[str] = []
        hypotheses: List[str] = []
        fallbacks = 0
        for b_row, ab_row in zip(test_b["rows"], test_ab["rows"]):
            if b_row["reference"] != ab_row["reference"]:
                raise RuntimeError("B-only 与 A+B 评测顺序不一致。")
            use_ab = float(b_row["confidence"]) < threshold
            references.append(str(b_row["reference"]))
            hypotheses.append(str(ab_row["hypothesis"] if use_ab else b_row["hypothesis"]))
            fallbacks += int(use_ab)
        metrics = compute_error_rates(references, hypotheses)
        results.append(
            {
                "target_train_fallback_rate": fallback_target,
                "confidence_threshold": threshold,
                "test_fallback_rate": fallbacks / max(1, len(references)),
                "cer": float(metrics["cer"]),
                "wer": float(metrics["wer"]),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 Python 环境没有可用 GPU。")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, test_records = load_records(args.label_cache)
    train_dataset = PaperV5EvalDataset(args.data_root, train_records, bin_size=2)
    test_dataset = PaperV5EvalDataset(args.data_root, test_records, bin_size=2)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_variable_paper_batch,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_variable_paper_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_variable_paper_batch,
    )

    requested = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = [name for name in requested if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    summary: Dict[str, Any] = {
        "protocol": {
            "train_records": len(train_records),
            "heldout_records": len(test_records),
            "test_used_for_model_selection": False,
            "geometry_source": "Dryad arrayGeometryMap; identical across 10 sessions",
            "channel_order": "1-96 lateral/anterior A; 97-192 medial/posterior B",
        },
        "variants": {},
    }
    cached_evaluations: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, GeometryCTCModel] = {}

    for name in requested:
        variant = VARIANTS[name]
        variant_dir = output_dir / name
        print(f"Starting {name}: {variant}", flush=True)
        model = train_variant(variant, args, train_loader, device, variant_dir)
        test_eval = evaluate_model(model, test_loader, device, variant.array_mode)
        train_eval = (
            evaluate_model(model, train_eval_loader, device, variant.array_mode)
            if name in {"ctc_graph_patch", "ctc_b_graph_patch"}
            else None
        )
        profile = model.base.hardware_profile()
        result = {
            "variant": variant.__dict__,
            "test_metrics": test_eval["metrics"],
            "test_mean_confidence": test_eval["mean_confidence"],
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "base_mac_per_20ms": float(profile["mac_per_20ms_step"]),
            "graph_adds_per_20ms": model.graph.hardware_adds_per_step(),
            "graph_multiplies_per_20ms": model.graph.hardware_multiplies_per_step(),
            "learned_graph_mix": model.graph.mix.detach().cpu().tolist(),
        }
        summary["variants"][name] = result
        cached_evaluations[name] = {"test": test_eval, "train": train_eval}
        models[name] = model
        (variant_dir / "test_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in test_eval["rows"]),
            encoding="utf-8",
        )
        print(f"Completed {name}: {result}", flush=True)

    if {"ctc_graph_patch", "ctc_b_graph_patch"}.issubset(cached_evaluations):
        fusion = conditional_fusion(
            cached_evaluations["ctc_b_graph_patch"]["train"],
            cached_evaluations["ctc_graph_patch"]["train"],
            cached_evaluations["ctc_b_graph_patch"]["test"],
            cached_evaluations["ctc_graph_patch"]["test"],
        )
        summary["b_first_conditional_fusion"] = fusion

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
