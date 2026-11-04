"""从 H144-v2 B2 出发做短阶段、结构不变的 N-best MWER 微调。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from brain_to_text_main_v5 import edit_distance, render_transcript
from compact_bci_model import CompactBrainToTextGRU
from paper_v5_pipeline import (
    CHAR_DEF,
    PaperBrainToTextV5,
    PaperV5EvalDataset,
    collate_variable_paper_batch,
    compute_paper_v5_loss,
)
from spatiotemporal_discriminability_atlas import split_train_internal
from train_compact_bci import (
    add_input_noise,
    autocast_context,
    compute_distillation_loss,
    evaluate_paper_v5_grid,
    load_frozen_label_cache,
    move_batch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def word_error(reference: str, hypothesis: str) -> float:
    reference_words = render_transcript(reference).split()
    hypothesis_words = render_transcript(hypothesis).split()
    return edit_distance(reference_words, hypothesis_words) / max(len(reference_words), 1)


def character_beams(
    char_log_prob: torch.Tensor,
    onset_log_prob: torch.Tensor,
    event_indices: Sequence[int],
    char_indices: Sequence[int],
    beam_width: int,
    char_topk: int,
) -> List[Tuple[str, torch.Tensor]]:
    if not event_indices:
        return [("", onset_log_prob.new_zeros(()))]
    # 排序只使用一次性搬到 CPU 的数值副本，保留的 tensor score 仍参与反向传播。
    # 这样可避免 beam 扩展中每个候选触发一次 GPU->CPU 同步。
    selected_char = char_log_prob[list(char_indices)].detach().float().cpu()
    selected_onset = onset_log_prob[list(event_indices)].detach().float().cpu()
    beams: List[Tuple[List[int], torch.Tensor, float]] = [
        ([], onset_log_prob.new_zeros(()), 0.0)
    ]
    for position, (event_index, char_index) in enumerate(zip(event_indices, char_indices)):
        top_values, top_index_tensor = torch.topk(
            selected_char[position], k=min(char_topk, char_log_prob.shape[-1])
        )
        top_indices = top_index_tensor.tolist()
        numeric_char = top_values.tolist()
        numeric_onset = float(selected_onset[position])
        expanded: List[Tuple[List[int], torch.Tensor, float]] = []
        for prefix, score, numeric_score in beams:
            for rank, class_index in enumerate(top_indices):
                expanded.append(
                    (
                        [*prefix, int(class_index)],
                        score
                        + char_log_prob[char_index, int(class_index)]
                        + 0.25 * onset_log_prob[event_index],
                        numeric_score + float(numeric_char[rank]) + 0.25 * numeric_onset,
                    )
                )
        beams = sorted(expanded, key=lambda item: item[2], reverse=True)[:beam_width]
    result: List[Tuple[str, torch.Tensor]] = []
    for classes, score, _numeric_score in beams:
        text = "".join(CHAR_DEF["charListAbbr"][index] for index in classes)
        # 长度归一化抑制跨 threshold 候选之间的短句偏置。
        result.append((text, score / max(len(classes), 1) ** 0.7))
    return result


def generate_nbest(
    logits: torch.Tensor,
    length: int,
    thresholds: Sequence[float],
    delays: Sequence[int],
    nbest: int,
    char_topk: int,
) -> List[Tuple[str, torch.Tensor]]:
    usable = logits[:length]
    char_log_prob = F.log_softmax(usable[:, :-1].float(), dim=-1)
    onset_prob = torch.sigmoid(usable[:, -1].float())
    onset_log_prob = torch.log(onset_prob.clamp_min(1.0e-6))
    detached_onset = onset_prob.detach()
    candidates: Dict[str, Tuple[torch.Tensor, float]] = {}
    for threshold in thresholds:
        crossings = torch.argwhere(
            (detached_onset[:-1] < threshold) & (detached_onset[1:] >= threshold)
        ).reshape(-1)
        event_indices = [int(value) + 1 for value in crossings.tolist()]
        for delay in delays:
            char_indices = [min(index + int(delay), length - 1) for index in event_indices]
            for text, score in character_beams(
                char_log_prob,
                onset_log_prob,
                event_indices,
                char_indices,
                nbest,
                char_topk,
            ):
                detached_score = float(score.detach().cpu())
                previous = candidates.get(text)
                if previous is None or detached_score > previous[1]:
                    candidates[text] = (score, detached_score)
    ranked = sorted(candidates.items(), key=lambda item: item[1][1], reverse=True)[:nbest]
    return [(text, score_and_value[0]) for text, score_and_value in ranked]


def mwer_objective(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    references: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: List[torch.Tensor] = []
    expected_values: List[float] = []
    oracle_values: List[float] = []
    candidate_counts: List[int] = []
    for row, length in enumerate(lengths.detach().cpu().tolist()):
        candidates = generate_nbest(
            logits[row],
            int(length),
            args.mwer_thresholds,
            args.mwer_delays,
            args.nbest,
            args.char_topk,
        )
        if not candidates:
            continue
        scores = torch.stack([score for _text, score in candidates])
        risks = torch.tensor(
            [word_error(str(references[row]), text) for text, _score in candidates],
            dtype=scores.dtype,
            device=scores.device,
        )
        probabilities = F.softmax(scores / args.mwer_score_temperature, dim=0)
        centered_risk = risks - risks.mean()
        losses.append(torch.sum(probabilities * centered_risk))
        expected_values.append(float(torch.sum(probabilities.detach() * risks).cpu()))
        oracle_values.append(float(risks.min().cpu()))
        candidate_counts.append(len(candidates))
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {"mwer_expected": 0.0, "mwer_oracle": 0.0, "nbest_count": 0.0}
    return torch.stack(losses).mean(), {
        "mwer_expected": float(np.mean(expected_values)),
        "mwer_oracle": float(np.mean(oracle_values)),
        "nbest_count": float(np.mean(candidate_counts)),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": model.export_config(),
            "hardware_profile": model.hardware_profile(),
            "validation_metrics": metrics,
            "train_config": vars(args),
            "source_checkpoint": args.student_checkpoint,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--label_cache", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--learning_rate_end", type=float, default=1.0e-5)
    parser.add_argument("--adam_eps", type=float, default=1.0e-8)
    parser.add_argument("--output_delay", type=int, default=50)
    parser.add_argument("--distill_char_weight", type=float, default=1.0)
    parser.add_argument("--distill_onset_weight", type=float, default=0.5)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--mwer_weight", type=float, default=0.1)
    parser.add_argument("--nbest", type=int, default=8)
    parser.add_argument("--char_topk", type=int, default=2)
    parser.add_argument("--mwer_thresholds", default="0.25,0.30,0.35")
    parser.add_argument("--mwer_delays", default="10,15,20")
    parser.add_argument("--mwer_score_temperature", type=float, default=1.0)
    parser.add_argument("--white_noise_sd", type=float, default=0.2)
    parser.add_argument("--constant_offset_sd", type=float, default=0.1)
    parser.add_argument("--random_walk_sd", type=float, default=0.005)
    parser.add_argument("--validate_every", type=int, default=2)
    parser.add_argument("--patience_validations", type=int, default=8)
    parser.add_argument("--decode_thresholds", default="0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--decode_delays", default="10,15,20,25")
    parser.add_argument("--amp", choices=("on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.mwer_thresholds = [float(value) for value in args.mwer_thresholds.split(",")]
    args.mwer_delays = [int(value) for value in args.mwer_delays.split(",")]
    args.decode_grid = True
    args.amp_enabled = args.amp == "on" and torch.cuda.is_available()
    set_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = load_frozen_label_cache(args.label_cache, args.data_root)
    official_train = [record for record in payload["records"] if record.split == "train"]
    fit_records, development_records, blocks = split_train_internal(official_train)
    train_dataset = PaperV5EvalDataset(
        payload["datasets_root"], fit_records, 2, training_steps_root=None
    )
    development_dataset = PaperV5EvalDataset(
        payload["datasets_root"], development_records, 2, training_steps_root=None
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_paper_batch,
        generator=generator,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_variable_paper_batch,
    )

    student_payload = torch.load(
        args.student_checkpoint, map_location="cpu", weights_only=False
    )
    model = CompactBrainToTextGRU(**student_payload["model_config"])
    model.load_state_dict(student_payload["model_state"])
    model.to(device)
    teacher_payload = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False
    )
    teacher = PaperBrainToTextV5(**teacher_payload["model_config"])
    teacher.load_state_dict(teacher_payload["model_state"])
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=args.adam_eps
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.learning_rate_end / args.learning_rate,
        total_iters=max(args.epochs, 1),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_enabled)
    history: List[Dict[str, object]] = []
    baseline = evaluate_paper_v5_grid(
        model, development_loader, device, args
    )
    best_metrics = dict(baseline)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    print(
        "C1 baseline | " + json.dumps(baseline, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Dict[str, float] = {
            "loss": 0.0,
            "supervised": 0.0,
            "char_kd": 0.0,
            "onset_kd": 0.0,
            "mwer": 0.0,
            "mwer_expected": 0.0,
            "mwer_oracle": 0.0,
            "nbest_count": 0.0,
        }
        steps = 0
        for batch in train_loader:
            moved = move_batch(batch, device)
            moved["inputs"] = add_input_noise(
                moved["inputs"],
                args.white_noise_sd,
                args.constant_offset_sd,
                args.random_walk_sd,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp_enabled):
                with torch.no_grad():
                    teacher_logits, _ = teacher(
                        moved["inputs"], moved["input_lengths"], moved["session_ids"]
                    )
                logits, _ = model(
                    moved["inputs"], moved["input_lengths"], moved["session_ids"]
                )
                supervised, _ = compute_paper_v5_loss(
                    logits,
                    moved["char_prob_target"],
                    moved["char_start_target"],
                    moved["err_weights"],
                    output_delay=args.output_delay,
                )
                char_kd, onset_kd, _ = compute_distillation_loss(
                    logits,
                    teacher_logits,
                    moved["err_weights"],
                    args.output_delay,
                    args.distill_temperature,
                )
                mwer, mwer_stats = mwer_objective(
                    logits,
                    moved["input_lengths"],
                    batch["transcripts"],
                    args,
                )
                loss = (
                    supervised
                    + args.distill_char_weight * char_kd
                    + args.distill_onset_weight * onset_kd
                    + args.mwer_weight * mwer
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            values = {
                "loss": float(loss.item()),
                "supervised": float(supervised.item()),
                "char_kd": float(char_kd.item()),
                "onset_kd": float(onset_kd.item()),
                "mwer": float(mwer.item()),
                **mwer_stats,
            }
            for key, value in values.items():
                totals[key] += float(value)
            steps += 1
        scheduler.step()
        train_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
        record: Dict[str, object] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": None,
        }
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            validation = evaluate_paper_v5_grid(
                model, development_loader, device, args
            )
            record["validation"] = validation
            key = (float(validation["wer"]), float(validation["cer"]))
            best_key = (float(best_metrics["wer"]), float(best_metrics["cer"]))
            if key < best_key:
                best_metrics = dict(validation)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
                save_checkpoint(
                    output / "best.pt", model, optimizer, epoch, validation, args
                )
            else:
                stale += 1
            print(
                f"epoch={epoch:03d} loss={train_metrics['loss']:.4f} "
                f"MWER={train_metrics['mwer']:.4f} "
                f"dev CER={validation['cer']:.4f} WER={validation['wer']:.4f}",
                flush=True,
            )
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if stale >= args.patience_validations:
            break

    model.load_state_dict(best_state)
    # 若 MWER 没有超过 epoch-0 B2，仍保存明确的基准回退点。
    if best_epoch == 0:
        save_checkpoint(output / "best.pt", model, optimizer, 0, best_metrics, args)
    summary = {
        "protocol": {
            "fit_records": len(fit_records),
            "development_records": len(development_records),
            "development_blocks": blocks,
            "official_heldout_signals_loaded": False,
        },
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best": best_metrics,
        "hardware": model.hardware_profile(),
        "mwer": {
            "weight": args.mwer_weight,
            "nbest": args.nbest,
            "thresholds": args.mwer_thresholds,
            "delays": args.mwer_delays,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
