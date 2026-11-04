"""B2 -> R0/R1/R2/R3 严格递归 W8A8/H16 QAT 与定点诊断。

R0：原始 B2 FP32；
R1：403 条 fit 数据校准后冻结的逐层 power-of-two scale；
R2：冻结网络参数，仅学习逐层 clipping/scale；
R3：严格回写 INT16 hidden 的 recurrent-aware 全模型 QAT。

官方 228 条 HeldOutBlocks 信号不会被加载。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compact_bci_model import CompactBrainToTextGRU
from paper_v5_pipeline import (
    PaperBrainToTextV5,
    PaperV5EvalDataset,
    PaperV5RealDataset,
    PaperV5SyntheticDataset,
    build_snippet_library,
    collate_variable_paper_batch,
    collect_word_list,
    compute_paper_v5_loss,
)
from recurrent_fixed_point_model_mp import PowerOfTwoQuantizer, RecurrentFixedPointH144
from spatiotemporal_discriminability_atlas import split_train_internal
from train_compact_bci import (
    add_input_noise,
    collate_training_batch,
    compute_distillation_loss,
    evaluate_paper_v5_grid,
    load_frozen_label_cache,
    move_batch,
    train_one_epoch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: RecurrentFixedPointH144,
    optimizer: torch.optim.Optimizer | None,
    stage: str,
    epoch: int,
    metrics: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "model_config": model.export_config(),
            "fixed_point_specification": model.fixed_point_specification(),
            "gate_precision": model.gate_precision_specification(),
            "stage": stage,
            "epoch": int(epoch),
            "validation_metrics": metrics,
            "source_checkpoint": args.student_checkpoint,
            "train_config": vars(args),
        },
        path,
    )


@torch.no_grad()
def calibrate_scales(
    model: RecurrentFixedPointH144,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    max_batches: int,
    margin: float,
) -> Tuple[int, Dict[str, int]]:
    """在 FP32 显式递归图上观察分布，再一次性冻结逐层 scale。"""

    model.eval()
    model.quantization_enabled = False
    model.set_observer_mode(True, True)
    batches = 0
    for batch in loader:
        moved = move_batch(batch, device)
        model(moved["inputs"], moved["input_lengths"], moved["session_ids"])
        batches += 1
        if batches >= max_batches:
            break
    model.set_observer_mode(False, False)
    exponents = model.freeze_observed_scales(margin)
    model.quantization_enabled = True
    return batches, exponents


def configure_parameter_training(
    model: RecurrentFixedPointH144,
    train_network: bool,
    train_scales: bool,
) -> List[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(train_network)
    model.set_scale_trainable(train_scales)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def scale_regularizer(
    model: RecurrentFixedPointH144,
    reference_exponents: Dict[str, float],
) -> torch.Tensor:
    penalty = next(model.parameters()).new_zeros(())
    for name, quantizer in model.quantizers().items():
        if name in reference_exponents and quantizer.log2_scale.requires_grad:
            target = quantizer.log2_scale.new_tensor(reference_exponents[name])
            penalty = penalty + (quantizer.log2_scale - target).square()
    return penalty


def train_scale_only_epoch(
    model: RecurrentFixedPointH144,
    fp_teacher: CompactBrainToTextGRU,
    loader: Iterable[Dict[str, object]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    reference_exponents: Dict[str, float],
) -> Dict[str, float]:
    model.train()
    fp_teacher.eval()
    totals = {"loss": 0.0, "char_kl": 0.0, "onset_mse": 0.0, "supervised": 0.0, "scale_reg": 0.0}
    steps = 0
    for batch_index, batch in enumerate(loader):
        moved = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits, _ = fp_teacher(
                moved["inputs"], moved["input_lengths"], moved["session_ids"]
            )
        logits, _ = model(
            moved["inputs"], moved["input_lengths"], moved["session_ids"]
        )
        char_kl, onset_mse, _ = compute_distillation_loss(
            logits,
            teacher_logits,
            moved["err_weights"],
            args.output_delay,
            1.0,
        )
        supervised, _ = compute_paper_v5_loss(
            logits,
            moved["char_prob_target"],
            moved["char_start_target"],
            moved["err_weights"],
            output_delay=args.output_delay,
        )
        regularizer = scale_regularizer(model, reference_exponents)
        loss = char_kl + 0.5 * onset_mse + 0.2 * supervised + args.scale_regularizer * regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()
        values = {
            "loss": loss,
            "char_kl": char_kl,
            "onset_mse": onset_mse,
            "supervised": supervised,
            "scale_reg": regularizer,
        }
        for key, value in values.items():
            totals[key] += float(value.detach().item())
        steps += 1
        if args.max_scale_batches > 0 and batch_index + 1 >= args.max_scale_batches:
            break
    return {key: value / max(steps, 1) for key, value in totals.items()}


def component_error(
    reference: torch.Tensor,
    quantized: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    max_steps = reference.shape[1]
    numerator = np.zeros(max_steps, dtype=np.float64)
    denominator = np.zeros(max_steps, dtype=np.float64)
    for row, length in enumerate(valid_lengths.detach().cpu().tolist()):
        length = min(int(length), max_steps)
        difference = (reference[row, :length] - quantized[row, :length]).double()
        baseline = reference[row, :length].double()
        reduce_dims = tuple(range(1, difference.ndim))
        numerator[:length] += difference.square().sum(dim=reduce_dims).detach().cpu().numpy()
        denominator[:length] += baseline.square().sum(dim=reduce_dims).detach().cpu().numpy()
    return numerator, denominator


@torch.no_grad()
def run_recurrent_diagnostics(
    model: RecurrentFixedPointH144,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    output_dir: Path,
    max_batches: int,
) -> Dict[str, object]:
    """比较相同 shadow weights 的 FP32 与严格定点递归轨迹。"""

    model.eval()
    accumulators: Dict[str, Dict[str, np.ndarray]] = {}
    clip_counts = {
        "gru1_hidden": [0, 0],
        "gru2_hidden": [0, 0],
        "gru1_reset": [0, 0],
        "gru1_update": [0, 0],
        "gru1_candidate": [0, 0],
        "gru2_reset": [0, 0],
        "gru2_update": [0, 0],
        "gru2_candidate": [0, 0],
        "output_logits": [0, 0],
    }
    batches = 0
    for batch in loader:
        moved = move_batch(batch, device)
        selected_activation_mode = model.activation_mode
        model.activation_mode = "exact"
        model.quantization_enabled = False
        model.set_observer_mode(False, True)
        float_logits, _lengths, float_trace = model(
            moved["inputs"], moved["input_lengths"], moved["session_ids"], return_trace=True
        )
        model.activation_mode = selected_activation_mode
        model.quantization_enabled = True
        model.set_observer_mode(False, False)
        quant_logits, _lengths, quant_trace = model(
            moved["inputs"], moved["input_lengths"], moved["session_ids"], return_trace=True
        )
        down_lengths = torch.div(
            moved["input_lengths"] + model.skip_len - 1,
            model.skip_len,
            rounding_mode="floor",
        ).clamp_min(1)
        components = {
            "gru1_hidden": (float_trace["gru1"]["hidden"], quant_trace["gru1"]["hidden"], moved["input_lengths"]),
            "gru1_reset": (float_trace["gru1"]["reset"], quant_trace["gru1"]["reset"], moved["input_lengths"]),
            "gru1_update": (float_trace["gru1"]["update"], quant_trace["gru1"]["update"], moved["input_lengths"]),
            "gru1_candidate": (float_trace["gru1"]["candidate"], quant_trace["gru1"]["candidate"], moved["input_lengths"]),
            "gru2_hidden": (float_trace["gru2"]["hidden"], quant_trace["gru2"]["hidden"], down_lengths),
            "gru2_reset": (float_trace["gru2"]["reset"], quant_trace["gru2"]["reset"], down_lengths),
            "gru2_update": (float_trace["gru2"]["update"], quant_trace["gru2"]["update"], down_lengths),
            "gru2_candidate": (float_trace["gru2"]["candidate"], quant_trace["gru2"]["candidate"], down_lengths),
            "character_logits": (float_logits[:, :, :-1], quant_logits[:, :, :-1], moved["input_lengths"]),
            "onset_logits": (float_logits[:, :, -1:], quant_logits[:, :, -1:], moved["input_lengths"]),
        }
        for name, (reference, quantized, valid_lengths) in components.items():
            numerator, denominator = component_error(reference, quantized, valid_lengths)
            if name not in accumulators:
                accumulators[name] = {"numerator": numerator, "denominator": denominator}
            else:
                old = accumulators[name]
                if len(old["numerator"]) < len(numerator):
                    extension = len(numerator) - len(old["numerator"])
                    old["numerator"] = np.pad(old["numerator"], (0, extension))
                    old["denominator"] = np.pad(old["denominator"], (0, extension))
                old["numerator"][: len(numerator)] += numerator
                old["denominator"][: len(denominator)] += denominator

        for layer_name, trace_key, quantizer in (
            ("gru1_hidden", "gru1", model.hidden1_q),
            ("gru2_hidden", "gru2", model.hidden2_q),
        ):
            pre_hidden = quant_trace[trace_key]["pre_hidden"]
            lengths = moved["input_lengths"] if trace_key == "gru1" else down_lengths
            clip = float(quantizer.qmax * quantizer.scale().detach().cpu().item())
            for row, length in enumerate(lengths.detach().cpu().tolist()):
                values = pre_hidden[row, : int(length)]
                clip_counts[layer_name][0] += int((values.abs() > clip).sum().item())
                clip_counts[layer_name][1] += int(values.numel())
        gate_quantizers = {
            "reset": model.reset_q,
            "update": model.update_q,
            "candidate": model.candidate_q,
        }
        for trace_key, lengths in (("gru1", moved["input_lengths"]), ("gru2", down_lengths)):
            for gate_name, quantizer in gate_quantizers.items():
                values = quant_trace[trace_key][f"{gate_name}_unquantized"]
                lower = float(quantizer.qmin * quantizer.scale().detach().cpu().item())
                upper = float(quantizer.qmax * quantizer.scale().detach().cpu().item())
                key = f"{trace_key}_{gate_name}"
                for row, length in enumerate(lengths.detach().cpu().tolist()):
                    selected = values[row, : int(length)]
                    clip_counts[key][0] += int(((selected < lower) | (selected > upper)).sum().item())
                    clip_counts[key][1] += int(selected.numel())
        logit_values = quant_trace["logits_pre_quant"]
        lower = float(model.logit_q.qmin * model.logit_q.scale().detach().cpu().item())
        upper = float(model.logit_q.qmax * model.logit_q.scale().detach().cpu().item())
        for row, length in enumerate(down_lengths.detach().cpu().tolist()):
            selected = logit_values[row, : int(length)]
            clip_counts["output_logits"][0] += int(
                ((selected < lower) | (selected > upper)).sum().item()
            )
            clip_counts["output_logits"][1] += int(selected.numel())
        batches += 1
        if max_batches > 0 and batches >= max_batches:
            break

    model.quantization_enabled = True
    model.set_observer_mode(False, False)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {"batches": batches, "components": {}, "clip_rates": {}}
    for name, values in accumulators.items():
        error = np.sqrt(values["numerator"] / np.maximum(values["denominator"], 1.0e-12))
        finite = np.isfinite(error)
        valid_error = error[finite]
        summary["components"][name] = {
            "mean_relative_l2": float(valid_error.mean()) if len(valid_error) else 0.0,
            "first_50_relative_l2": float(valid_error[:50].mean()) if len(valid_error) else 0.0,
            "last_50_relative_l2": float(valid_error[-50:].mean()) if len(valid_error) else 0.0,
            "time_steps": int(len(valid_error)),
        }
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_index", "relative_l2"])
            for index, value in enumerate(error.tolist()):
                writer.writerow([index, value])
    for name, (clipped, total) in clip_counts.items():
        summary["clip_rates"][name] = {
            "clipped": clipped,
            "total": total,
            "rate": clipped / max(total, 1),
        }

    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        groups = [
            (axes[0, 0], ["gru1_hidden", "gru2_hidden"], "Hidden relative error"),
            (axes[0, 1], ["gru1_reset", "gru1_update", "gru1_candidate"], "GRU1 gate error"),
            (axes[1, 0], ["gru2_reset", "gru2_update", "gru2_candidate"], "GRU2 gate error"),
            (axes[1, 1], ["character_logits", "onset_logits"], "Output logit error"),
        ]
        for axis, names, title in groups:
            for name in names:
                values = accumulators[name]
                error = np.sqrt(values["numerator"] / np.maximum(values["denominator"], 1.0e-12))
                axis.plot(error, label=name)
            axis.set_title(title)
            axis.set_xlabel("time index")
            axis.set_ylabel("relative L2")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "recurrent_quantization_error.png", dpi=180)
        plt.close(figure)
    except Exception as exc:  # 图不是数值实验的必要条件。
        summary["plot_error"] = str(exc)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--label_cache", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output_delay", type=int, default=50)
    parser.add_argument("--calibration_batches", type=int, default=16)
    parser.add_argument("--calibration_batch_size", type=int, default=8)
    parser.add_argument("--calibration_margin", type=float, default=1.05)
    parser.add_argument("--scale_epochs", type=int, default=6)
    parser.add_argument("--scale_learning_rate", type=float, default=0.02)
    parser.add_argument("--scale_regularizer", type=float, default=0.01)
    parser.add_argument("--max_scale_batches", type=int, default=0)
    parser.add_argument("--qat_epochs", type=int, default=30)
    parser.add_argument("--qat_learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--qat_learning_rate_end", type=float, default=1.0e-5)
    parser.add_argument("--batch_size_real", type=int, default=12)
    parser.add_argument("--batch_size_synth", type=int, default=8)
    parser.add_argument("--synthetic_samples_per_epoch", type=int, default=128)
    parser.add_argument("--time_steps", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--validate_every", type=int, default=2)
    parser.add_argument("--patience_validations", type=int, default=8)
    parser.add_argument("--diagnostic_batches", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--smoke_development_records", type=int, default=0)
    parser.add_argument("--distill_char_weight", type=float, default=1.0)
    parser.add_argument("--distill_onset_weight", type=float, default=0.5)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--white_noise_sd", type=float, default=0.2)
    parser.add_argument("--constant_offset_sd", type=float, default=0.1)
    parser.add_argument("--random_walk_sd", type=float, default=0.005)
    parser.add_argument("--decode_thresholds", default="0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--decode_delays", default="10,15,20,25")
    # Gate-wise activation precision. Existing W8 per-output-channel weights and
    # INT16 pre-activation path are retained for compatibility with the current RTL.
    parser.add_argument("--reset_gate_bits", type=int, choices=(8, 12, 16), default=16)
    parser.add_argument("--update_gate_bits", type=int, choices=(8, 12, 16), default=16)
    parser.add_argument("--candidate_gate_bits", type=int, choices=(8, 12, 16), default=16)
    args = parser.parse_args()
    args.decode_grid = True
    args.amp_enabled = False
    args.distill_warmup_epochs = args.qat_epochs
    args.distill_every = 1
    args.l2_scale = 1.0e-5
    args.grad_clip = 10.0
    args.train_log_every = 0
    set_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = load_frozen_label_cache(args.label_cache, args.data_root)
    official_train = [record for record in payload["records"] if record.split == "train"]
    fit_records, development_records, blocks = split_train_internal(official_train)
    if args.smoke_development_records > 0:
        development_records = development_records[: args.smoke_development_records]
    fit_payload = {**payload, "records": list(fit_records)}
    calibration_dataset = PaperV5EvalDataset(
        payload["datasets_root"], fit_records, 2, training_steps_root=None
    )
    development_dataset = PaperV5EvalDataset(
        payload["datasets_root"], development_records, 2, training_steps_root=None
    )
    real_dataset = PaperV5RealDataset(
        payload["datasets_root"], fit_records, args.time_steps, 2, training_steps_root=None
    )
    snippet_library = build_snippet_library(
        payload["datasets_root"], fit_payload, training_steps_root=None
    )
    words, rare_word_indices = collect_word_list(fit_records)
    synthetic_dataset = PaperV5SyntheticDataset(
        snippet_library,
        words,
        rare_word_indices,
        args.time_steps,
        2,
        args.synthetic_samples_per_epoch,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False,
        collate_fn=collate_variable_paper_batch,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_variable_paper_batch,
    )
    real_loader = DataLoader(
        real_dataset,
        batch_size=args.batch_size_real,
        shuffle=True,
        collate_fn=collate_training_batch,
    )
    synthetic_loader = DataLoader(
        synthetic_dataset,
        batch_size=args.batch_size_synth,
        shuffle=True,
        collate_fn=collate_training_batch,
    )

    source = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    fp_model = CompactBrainToTextGRU(**source["model_config"])
    fp_model.load_state_dict(source["model_state"])
    fp_model.to(device).eval()
    model = RecurrentFixedPointH144(**source["model_config"])
    missing, unexpected = model.load_state_dict(source["model_state"], strict=False)
    if unexpected or any("_q." not in name for name in missing):
        raise RuntimeError(f"Fixed-point state mismatch: missing={missing}, unexpected={unexpected}")
    gate_precision = model.set_gate_precision(
        reset_bits=args.reset_gate_bits,
        update_bits=args.update_gate_bits,
        candidate_bits=args.candidate_gate_bits,
    )
    model.to(device)

    r0 = evaluate_paper_v5_grid(fp_model, development_loader, device, args)
    calibrated_batches, calibrated_exponents = calibrate_scales(
        model,
        calibration_loader,
        device,
        args.calibration_batches,
        args.calibration_margin,
    )
    model.eval()
    r1 = evaluate_paper_v5_grid(model, development_loader, device, args)
    save_checkpoint(output / "R1_fixed_scale.pt", model, None, "R1", 0, r1, args)
    print("R0 " + json.dumps(r0, ensure_ascii=False), flush=True)
    print("R1 " + json.dumps(r1, ensure_ascii=False), flush=True)

    reference_exponents = {
        name: float(quantizer.log2_scale.detach().cpu().item())
        for name, quantizer in model.quantizers().items()
    }
    scale_parameters = configure_parameter_training(model, False, True)
    scale_optimizer = torch.optim.Adam(scale_parameters, lr=args.scale_learning_rate)
    r2_best = dict(r1)
    r2_best_state = copy.deepcopy(model.state_dict())
    r2_best_epoch = 0
    r2_history: List[Dict[str, object]] = []
    for epoch in range(1, args.scale_epochs + 1):
        train_metrics = train_scale_only_epoch(
            model,
            fp_model,
            real_loader,
            scale_optimizer,
            device,
            args,
            reference_exponents,
        )
        validation = evaluate_paper_v5_grid(model, development_loader, device, args)
        if (float(validation["wer"]), float(validation["cer"])) < (
            float(r2_best["wer"]),
            float(r2_best["cer"]),
        ):
            r2_best = dict(validation)
            r2_best_epoch = epoch
            r2_best_state = copy.deepcopy(model.state_dict())
        r2_history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        print(
            f"R2 epoch={epoch:02d} loss={train_metrics['loss']:.5f} "
            f"CER={validation['cer']:.5f} WER={validation['wer']:.5f}",
            flush=True,
        )
    model.load_state_dict(r2_best_state)
    save_checkpoint(
        output / "R2_learned_scale.pt",
        model,
        scale_optimizer,
        "R2",
        r2_best_epoch,
        r2_best,
        args,
    )
    (output / "R2_history.json").write_text(
        json.dumps(r2_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    teacher_payload = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = PaperBrainToTextV5(**teacher_payload["model_config"])
    teacher.load_state_dict(teacher_payload["model_state"])
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    qat_parameters = configure_parameter_training(model, True, True)
    optimizer = torch.optim.Adam(qat_parameters, lr=args.qat_learning_rate, eps=1.0e-8)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.qat_learning_rate_end / args.qat_learning_rate,
        total_iters=max(args.qat_epochs, 1),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    r3_best = dict(r2_best)
    r3_best_state = copy.deepcopy(model.state_dict())
    r3_best_epoch = 0
    r3_history: List[Dict[str, object]] = []
    stale = 0
    for epoch in range(1, args.qat_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            teacher,
            optimizer,
            real_loader,
            synthetic_loader,
            device,
            args,
            scaler,
            epoch,
        )
        scheduler.step()
        validation = None
        if epoch % args.validate_every == 0 or epoch == args.qat_epochs:
            validation = evaluate_paper_v5_grid(model, development_loader, device, args)
            if (float(validation["wer"]), float(validation["cer"])) < (
                float(r3_best["wer"]),
                float(r3_best["cer"]),
            ):
                r3_best = dict(validation)
                r3_best_epoch = epoch
                r3_best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            print(
                f"R3 epoch={epoch:03d} loss={train_metrics['loss']:.5f} "
                f"CER={validation['cer']:.5f} WER={validation['wer']:.5f}",
                flush=True,
            )
        r3_history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        (output / "R3_history.json").write_text(
            json.dumps(r3_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if stale >= args.patience_validations:
            break
    model.load_state_dict(r3_best_state)
    save_checkpoint(
        output / "R3_recurrent_qat.pt",
        model,
        optimizer,
        "R3",
        r3_best_epoch,
        r3_best,
        args,
    )

    nonlinear_results: Dict[str, Dict[str, object]] = {}
    variants = [
        ("exact", "exact", 0),
        ("lut256", "lut", 256),
        ("lut512", "lut", 512),
        ("pwl8", "pwl", 8),
        ("pwl16", "pwl", 16),
    ]
    for name, mode, size in variants:
        model.activation_mode = mode
        if mode == "lut":
            model.lut_entries = size
        if mode == "pwl":
            model.pwl_segments = size
        nonlinear_results[name] = evaluate_paper_v5_grid(
            model, development_loader, device, args
        )
        print(name + " " + json.dumps(nonlinear_results[name], ensure_ascii=False), flush=True)
    exact_wer = float(nonlinear_results["exact"]["wer"])
    acceptable_lut = [
        name
        for name in ("lut256", "lut512")
        if float(nonlinear_results[name]["wer"]) - exact_wer <= 0.002
    ]
    selected_nonlinear = next(
        (name for name in ("lut256", "lut512") if name in acceptable_lut),
        None,
    )
    if selected_nonlinear is None:
        selected_nonlinear = min(
            ("lut256", "lut512"),
            key=lambda name: (
                float(nonlinear_results[name]["wer"]),
                float(nonlinear_results[name]["cer"]),
            ),
        )
    model.activation_mode = "lut"
    model.lut_entries = int(selected_nonlinear[3:])

    diagnostics = run_recurrent_diagnostics(
        model,
        development_loader,
        device,
        output / "diagnostics",
        args.diagnostic_batches,
    )
    final_checkpoint = output / "fpga_candidate.pt"
    selected_metrics = nonlinear_results[selected_nonlinear]
    save_checkpoint(
        final_checkpoint,
        model,
        optimizer,
        "FPGA_candidate",
        r3_best_epoch,
        selected_metrics,
        args,
    )
    summary = {
        "protocol": {
            "fit_records": len(fit_records),
            "development_records": len(development_records),
            "development_blocks": blocks,
            "official_heldout_signals_loaded": False,
            "calibration_batches_from_fit_only": calibrated_batches,
        },
        "R0_fp32": r0,
        "R1_fixed_scale": r1,
        "R1_scale_exponents": calibrated_exponents,
        "R2_best_epoch": r2_best_epoch,
        "R2_learned_scale": r2_best,
        "R3_best_epoch": r3_best_epoch,
        "R3_recurrent_qat": r3_best,
        "nonlinear_results": nonlinear_results,
        "selected_nonlinear": selected_nonlinear,
        "selected_metrics": selected_metrics,
        "fixed_point_specification": model.fixed_point_specification(),
        "gate_precision_experiment": gate_precision,
        "diagnostics": diagnostics,
        "hardware": model.hardware_profile(),
        "checkpoints": {
            "R1": str(output / "R1_fixed_scale.pt"),
            "R2": str(output / "R2_learned_scale.pt"),
            "R3": str(output / "R3_recurrent_qat.pt"),
            "FPGA_candidate": str(final_checkpoint),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
