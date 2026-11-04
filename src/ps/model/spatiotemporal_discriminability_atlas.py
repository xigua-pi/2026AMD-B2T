"""构建 Utah 空间位置 × 时间尺度 × 任务的训练内辨识性图谱。

最终 HeldOutBlocks 完全不参与图谱评分或状态选择。官方 train 记录按每个
session 的最后一个训练 block 再切出内部开发集，用于衡量跨 block 稳定性。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter
from scipy.stats import rankdata, spearmanr

from geometry_ctc_experiment import DRYAD_GEOMETRY, build_neighbor_matrix, load_records
from paper_v5_pipeline import N_CHAR_CLASSES, PaperV5EvalDataset, SentenceLabelRecord


SPATIAL_NAMES = ("raw", "neighbor_mean", "local_contrast")
TEMPORAL_NAMES = ("iir", "delay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="Datasets")
    parser.add_argument("--label_cache", default="paper_v5_label_cache.pt")
    parser.add_argument("--output_dir", default="artifacts/spatiotemporal_atlas")
    parser.add_argument("--scales_ms", default="20,40,80,160,320,640,1280")
    parser.add_argument("--step_ms", type=int, default=20)
    parser.add_argument("--output_delay", type=int, default=50)
    return parser.parse_args()


def split_train_internal(
    records: Sequence[SentenceLabelRecord],
) -> Tuple[List[SentenceLabelRecord], List[SentenceLabelRecord], Dict[str, int]]:
    fit: List[SentenceLabelRecord] = []
    development: List[SentenceLabelRecord] = []
    held_blocks: Dict[str, int] = {}
    sessions = sorted({record.session_name for record in records})
    for session in sessions:
        session_records = [record for record in records if record.session_name == session]
        blocks = sorted({record.block_num for record in session_records})
        held_block = blocks[-1]
        held_blocks[session] = int(held_block)
        for record in session_records:
            (development if record.block_num == held_block else fit).append(record)
    return fit, development, held_blocks


def channel_positions() -> Dict[int, Tuple[int, int]]:
    result: Dict[int, Tuple[int, int]] = {}
    for row, col in np.argwhere(DRYAD_GEOMETRY > 0):
        result[int(DRYAD_GEOMETRY[row, col]) - 1] = (int(row), int(col))
    return result


def spatial_features(signal: np.ndarray, neighbor: np.ndarray) -> np.ndarray:
    neighbor_mean = signal @ neighbor.T
    return np.concatenate([signal, neighbor_mean, signal - neighbor_mean], axis=1).astype(
        np.float32
    )


def multiscale_states(base: np.ndarray, scale_steps: Sequence[int]) -> np.ndarray:
    states = []
    for steps in scale_steps:
        alpha = 1.0 / float(steps)
        if steps == 1:
            state = base
        else:
            # y[t] = (1-alpha)y[t-1] + alpha*x[t]，与移位式 IIR 状态一致。
            state = lfilter([alpha], [1.0, -(1.0 - alpha)], base, axis=0).astype(
                np.float32
            )
        states.append(state)
        # 纯移位状态保留轨迹细节；与 IIR 的低通记忆形成可比较的两种硬件基元。
        delayed = np.zeros_like(base)
        if steps < len(base):
            delayed[steps:] = base[:-steps]
        states.append(delayed)
    return np.concatenate(states, axis=1)


@dataclass
class Statistics:
    char_count: np.ndarray
    char_sum: np.ndarray
    char_sumsq: np.ndarray
    start_count: np.ndarray
    start_sum: np.ndarray
    start_sumsq: np.ndarray
    space_count: np.ndarray
    space_sum: np.ndarray
    space_sumsq: np.ndarray


def empty_statistics(n_features: int) -> Statistics:
    return Statistics(
        char_count=np.zeros((N_CHAR_CLASSES,), dtype=np.float64),
        char_sum=np.zeros((N_CHAR_CLASSES, n_features), dtype=np.float64),
        char_sumsq=np.zeros((N_CHAR_CLASSES, n_features), dtype=np.float64),
        start_count=np.zeros((2,), dtype=np.float64),
        start_sum=np.zeros((2, n_features), dtype=np.float64),
        start_sumsq=np.zeros((2, n_features), dtype=np.float64),
        space_count=np.zeros((2,), dtype=np.float64),
        space_sum=np.zeros((2, n_features), dtype=np.float64),
        space_sumsq=np.zeros((2, n_features), dtype=np.float64),
    )


def accumulate_binary(
    count: np.ndarray,
    total: np.ndarray,
    total_sq: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    for label in (0, 1):
        selected = features[labels == label]
        if len(selected) == 0:
            continue
        count[label] += len(selected)
        total[label] += selected.sum(axis=0, dtype=np.float64)
        total_sq[label] += np.square(selected, dtype=np.float64).sum(axis=0)


def collect_statistics(
    dataset: PaperV5EvalDataset,
    neighbor: np.ndarray,
    scale_steps: Sequence[int],
    output_delay: int,
    label: str,
) -> Statistics:
    n_features = len(scale_steps) * len(TEMPORAL_NAMES) * len(SPATIAL_NAMES) * 192
    stats = empty_statistics(n_features)
    for index in range(len(dataset)):
        item = dataset[index]
        base = spatial_features(item["inputs"].numpy(), neighbor)
        state = multiscale_states(base, scale_steps)
        if len(state) <= output_delay:
            continue
        features = state[output_delay:]
        char_prob = item["char_prob_target"].numpy()[:-output_delay]
        start = item["char_start_target"].numpy()[:-output_delay]
        valid = item["err_weights"].numpy()[:-output_delay] > 0.5
        features = features[valid]
        char_prob = char_prob[valid]
        start = start[valid]

        char_active = char_prob.sum(axis=1) > 0.25
        char_labels = np.argmax(char_prob[char_active], axis=1)
        char_features = features[char_active]
        for class_index in np.unique(char_labels):
            selected = char_features[char_labels == class_index]
            stats.char_count[class_index] += len(selected)
            stats.char_sum[class_index] += selected.sum(axis=0, dtype=np.float64)
            stats.char_sumsq[class_index] += np.square(selected, dtype=np.float64).sum(axis=0)

        accumulate_binary(
            stats.start_count,
            stats.start_sum,
            stats.start_sumsq,
            features,
            (start > 0.10).astype(np.int64),
        )
        if len(char_features):
            accumulate_binary(
                stats.space_count,
                stats.space_sum,
                stats.space_sumsq,
                char_features,
                (char_labels == 26).astype(np.int64),
            )
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"atlas {label}: {index + 1}/{len(dataset)}", flush=True)
    return stats


def safe_mean_var(count: np.ndarray, total: np.ndarray, total_sq: np.ndarray):
    denominator = np.maximum(count[:, None], 1.0)
    mean = total / denominator
    variance = np.maximum(total_sq / denominator - mean * mean, 1.0e-8)
    return mean, variance


def multiclass_scores(stats: Statistics):
    mean, variance = safe_mean_var(stats.char_count, stats.char_sum, stats.char_sumsq)
    total_count = max(float(stats.char_count.sum()), 1.0)
    overall = stats.char_sum.sum(axis=0) / total_count
    between = (
        stats.char_count[:, None] * np.square(mean - overall[None, :])
    ).sum(axis=0) / total_count
    within = (stats.char_count[:, None] * variance).sum(axis=0) / total_count
    global_fisher = between / (within + 1.0e-8)

    per_class = np.zeros_like(mean)
    total_sum = stats.char_sum.sum(axis=0)
    total_sq = stats.char_sumsq.sum(axis=0)
    for class_index in range(N_CHAR_CLASSES):
        positive_count = max(float(stats.char_count[class_index]), 1.0)
        negative_count = max(total_count - positive_count, 1.0)
        positive_mean = stats.char_sum[class_index] / positive_count
        negative_sum = total_sum - stats.char_sum[class_index]
        negative_mean = negative_sum / negative_count
        positive_var = np.maximum(
            stats.char_sumsq[class_index] / positive_count - positive_mean**2, 1.0e-8
        )
        negative_var = np.maximum(
            (total_sq - stats.char_sumsq[class_index]) / negative_count - negative_mean**2,
            1.0e-8,
        )
        per_class[class_index] = np.square(positive_mean - negative_mean) / (
            positive_var + negative_var + 1.0e-8
        )
    return global_fisher, per_class


def binary_score(count: np.ndarray, total: np.ndarray, total_sq: np.ndarray) -> np.ndarray:
    mean, variance = safe_mean_var(count, total, total_sq)
    return np.square(mean[1] - mean[0]) / (variance[0] + variance[1] + 1.0e-8)


def percentile(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def stable_geometric(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(left, 0.0) * np.maximum(right, 0.0))


def decode_feature_index(index: int, scales_ms: Sequence[int]) -> Dict[str, object]:
    per_temporal = len(SPATIAL_NAMES) * 192
    per_scale = len(TEMPORAL_NAMES) * per_temporal
    scale_index, offset = divmod(index, per_scale)
    temporal_index, offset = divmod(offset, per_temporal)
    spatial_index, global_channel = divmod(offset, 192)
    local_channel = global_channel % 96
    row, col = channel_positions()[local_channel]
    return {
        "feature_index": int(index),
        "scale_index": int(scale_index),
        "scale_ms": int(scales_ms[scale_index]),
        "temporal_type": TEMPORAL_NAMES[temporal_index],
        "spatial_type": SPATIAL_NAMES[spatial_index],
        "dryad_channel": int(global_channel + 1),
        "array": "A_lateral_anterior" if global_channel < 96 else "B_medial_posterior",
        "local_channel": int(local_channel + 1),
        "grid_row": int(row + 1),
        "grid_col": int(col + 1),
    }


def greedy_hardware_selection(
    base_score: np.ndarray,
    scales_ms: Sequence[int],
    budget: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """在辨识性之外抑制结构冗余，并奖励可复用的空间预处理。

    这里不用最终留出集估计相关性，而用确定性的 Utah 邻接、空间算子和
    时间尺度接近程度作为冗余代理，因此不会把测试标签泄漏进结构选择。
    """

    metadata = [decode_feature_index(index, scales_ms) for index in range(len(base_score))]
    positions = np.asarray(
        [[int(item["grid_row"]), int(item["grid_col"])] for item in metadata],
        dtype=np.int16,
    )
    channels = np.asarray([int(item["dryad_channel"]) for item in metadata])
    arrays = np.asarray([str(item["array"]) for item in metadata])
    spatial = np.asarray([str(item["spatial_type"]) for item in metadata])
    temporal_type = np.asarray([str(item["temporal_type"]) for item in metadata])
    scales = np.asarray([int(item["scale_ms"]) for item in metadata], dtype=np.float64)

    selected: List[int] = []
    selected_mask = np.zeros(len(base_score), dtype=bool)
    max_redundancy = np.zeros(len(base_score), dtype=np.float64)
    selection_score = np.full(len(base_score), -np.inf, dtype=np.float64)
    used_spatial_sources = set()
    for _ in range(min(int(budget), len(base_score))):
        # 同一电极/算子的多个时间尺度可共享空间预处理，给一个很小的实现奖励。
        reuse = np.asarray(
            [
                1.0
                if (int(channels[i]), str(spatial[i]), str(temporal_type[i]))
                in used_spatial_sources
                else 0.0
                for i in range(len(base_score))
            ],
            dtype=np.float64,
        )
        candidate_score = base_score - 0.18 * max_redundancy + 0.025 * reuse
        candidate_score[selected_mask] = -np.inf
        chosen = int(np.argmax(candidate_score))
        selected.append(chosen)
        selected_mask[chosen] = True
        selection_score[chosen] = float(candidate_score[chosen])
        used_spatial_sources.add(
            (int(channels[chosen]), str(spatial[chosen]), str(temporal_type[chosen]))
        )

        log_scale_gap = np.abs(np.log2(scales / scales[chosen]))
        temporal_similarity = np.exp(-log_scale_gap)
        same_channel = channels == channels[chosen]
        same_spatial = spatial == spatial[chosen]
        same_temporal = temporal_type == temporal_type[chosen]
        same_array = arrays == arrays[chosen]
        manhattan = np.abs(positions[:, 0] - positions[chosen, 0]) + np.abs(
            positions[:, 1] - positions[chosen, 1]
        )
        redundancy = np.zeros(len(base_score), dtype=np.float64)
        same_basis = same_channel & same_spatial & same_temporal
        redundancy[same_basis] = 0.95 * temporal_similarity[
            same_basis
        ]
        alternate_basis = same_channel & same_spatial & ~same_temporal
        redundancy[alternate_basis] = np.maximum(
            redundancy[alternate_basis],
            0.45 * temporal_similarity[alternate_basis],
        )
        different_spatial = same_channel & ~same_spatial
        redundancy[different_spatial] = np.maximum(
            redundancy[different_spatial],
            0.35 * temporal_similarity[different_spatial],
        )
        adjacent = same_array & (manhattan == 1) & same_spatial & same_temporal
        redundancy[adjacent] = np.maximum(
            redundancy[adjacent], 0.20 * temporal_similarity[adjacent]
        )
        max_redundancy = np.maximum(max_redundancy, redundancy)
    return np.asarray(selected, dtype=np.int64), selection_score


def main() -> None:
    args = parse_args()
    scales_ms = [int(value) for value in args.scales_ms.split(",") if value.strip()]
    if any(value % args.step_ms for value in scales_ms):
        raise ValueError("所有时间尺度必须是 step_ms 的整数倍。")
    scale_steps = [value // args.step_ms for value in scales_ms]
    train_records, test_records = load_records(args.label_cache)
    fit_records, development_records, held_blocks = split_train_internal(train_records)
    neighbor = build_neighbor_matrix()[0].numpy().astype(np.float32)
    fit_dataset = PaperV5EvalDataset(args.data_root, fit_records, bin_size=2)
    development_dataset = PaperV5EvalDataset(
        args.data_root, development_records, bin_size=2
    )
    fit_stats = collect_statistics(
        fit_dataset, neighbor, scale_steps, args.output_delay, "fit"
    )
    dev_stats = collect_statistics(
        development_dataset, neighbor, scale_steps, args.output_delay, "development"
    )

    fit_global, fit_class = multiclass_scores(fit_stats)
    dev_global, dev_class = multiclass_scores(dev_stats)
    fit_start = binary_score(
        fit_stats.start_count, fit_stats.start_sum, fit_stats.start_sumsq
    )
    dev_start = binary_score(dev_stats.start_count, dev_stats.start_sum, dev_stats.start_sumsq)
    fit_space = binary_score(
        fit_stats.space_count, fit_stats.space_sum, fit_stats.space_sumsq
    )
    dev_space = binary_score(dev_stats.space_count, dev_stats.space_sum, dev_stats.space_sumsq)

    stable_global = stable_geometric(fit_global, dev_global)
    stable_class = stable_geometric(fit_class, dev_class)
    stable_start = stable_geometric(fit_start, dev_start)
    stable_space = stable_geometric(fit_space, dev_space)
    class_percentiles = np.stack(
        [percentile(stable_class[class_index]) for class_index in range(N_CHAR_CLASSES)],
        axis=0,
    )
    class_max = stable_class.max(axis=0)
    class_breadth = class_percentiles.mean(axis=0)
    spatial_cost = np.repeat(np.asarray([1.0, 1.15, 1.20]), 192)
    cost_parts = []
    for scale_ms in scales_ms:
        cost_parts.append(spatial_cost)
        # 延迟线的存储随时间跨度增加；IIR 每个尺度始终只需一个状态寄存器。
        delay_factor = 1.0 + 0.15 * np.log2(max(scale_ms / args.step_ms, 1.0))
        cost_parts.append(spatial_cost * delay_factor)
    cost = np.concatenate(cost_parts)
    combined = (
        0.30 * percentile(stable_global)
        + 0.25 * percentile(class_max)
        + 0.20 * class_breadth
        + 0.15 * percentile(stable_start)
        + 0.10 * percentile(stable_space)
    ) / cost
    selected, selection_score = greedy_hardware_selection(
        combined, scales_ms, budget=min(192, len(combined))
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(combined)):
        row = decode_feature_index(index, scales_ms)
        row.update(
            {
                "combined_score": float(combined[index]),
                "fit_char_fisher": float(fit_global[index]),
                "development_char_fisher": float(dev_global[index]),
                "stable_char_fisher": float(stable_global[index]),
                "stable_start_score": float(stable_start[index]),
                "stable_space_score": float(stable_space[index]),
                "max_class_score": float(class_max[index]),
                "class_breadth_percentile": float(class_breadth[index]),
                "selection_score": float(selection_score[index]),
                "rank": int(np.flatnonzero(selected == index)[0] + 1)
                if np.any(selected == index)
                else 0,
            }
        )
        rows.append(row)
    fieldnames = list(rows[0])
    with (output / "atlas.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    top_per_class = {}
    for class_index in range(N_CHAR_CLASSES):
        indices = np.argsort(-stable_class[class_index])[:10]
        top_per_class[str(class_index)] = [
            {
                **decode_feature_index(int(index), scales_ms),
                "score": float(stable_class[class_index, index]),
            }
            for index in indices
        ]

    summary = {
        "protocol": {
            "fit_records": len(fit_records),
            "development_records": len(development_records),
            "final_heldout_records_unused": len(test_records),
            "internal_development_blocks": held_blocks,
            "model_step_ms": args.step_ms,
            "scales_ms": scales_ms,
            "output_delay_steps": args.output_delay,
            "candidate_states": int(len(combined)),
        },
        "stability": {
            "char_global_spearman": float(spearmanr(fit_global, dev_global).statistic),
            "start_spearman": float(spearmanr(fit_start, dev_start).statistic),
            "space_spearman": float(spearmanr(fit_space, dev_space).statistic),
        },
        "selected_feature_indices": selected.tolist(),
        "top_32": [rows[int(index)] for index in selected[:32]],
        "top_per_class": top_per_class,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"protocol": summary["protocol"], "stability": summary["stability"], "top_5": summary["top_32"][:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
