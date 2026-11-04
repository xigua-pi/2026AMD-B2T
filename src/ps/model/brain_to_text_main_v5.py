from __future__ import annotations

import argparse
import math
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, Subset


VOCAB = "abcdefghijklmnopqrstuvwxyz>,'~?"
BLANK_INDEX = len(VOCAB)
DISPLAY_TRANSLATION = str.maketrans({">": " ", "~": "."})


def unwrap_mat_string(value: object) -> str:
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    return str(value)


def normalize_transcript(text: str) -> str:
    text = text.lower().strip()
    replacements = {
        " ": ">",
        ".": "~",
        "!": "~",
        "’": "'",
        "`": "'",
        '"': "",
        ":": "",
        ";": "",
        "-": "",
        "_": "",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return "".join(ch for ch in text if ch in VOCAB)


def render_transcript(text: str) -> str:
    rendered = text.translate(DISPLAY_TRANSLATION)
    rendered = rendered.replace(" ,", ",")
    rendered = rendered.replace(" ?", "?")
    rendered = rendered.replace(" '", "'")
    rendered = rendered.replace(" .", ".")
    return " ".join(rendered.split())


def encode_transcript(text: str) -> List[int]:
    return [VOCAB.index(ch) for ch in text]


def decode_indices(indices: Sequence[int]) -> str:
    return "".join(VOCAB[idx] for idx in indices)


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    table = np.zeros((rows, cols), dtype=np.int32)
    table[:, 0] = np.arange(rows)
    table[0, :] = np.arange(cols)

    for i in range(1, rows):
        for j in range(1, cols):
            substitution = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            table[i, j] = min(
                table[i - 1, j] + 1,
                table[i, j - 1] + 1,
                table[i - 1, j - 1] + substitution,
            )

    return int(table[-1, -1])


def compute_error_rates(references: Sequence[str], hypotheses: Sequence[str]) -> Dict[str, float]:
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
        "cer": char_errors / max(1, char_total),
        "wer": word_errors / max(1, word_total),
    }


def logsumexp_pair(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_greedy_decode(log_probs: torch.Tensor, blank_index: int = BLANK_INDEX) -> str:
    best_path = torch.argmax(log_probs, dim=-1).tolist()
    collapsed: List[int] = []
    prev = None
    for idx in best_path:
        if idx == blank_index:
            prev = idx
            continue
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return decode_indices(collapsed)


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam_width: int = 8,
    blank_index: int = BLANK_INDEX,
    prune_topk: int = 12,
) -> str:
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {tuple(): (0.0, -math.inf)}

    for time_index in range(log_probs.shape[0]):
        step = log_probs[time_index]
        topk = min(prune_topk, step.shape[0])
        top_indices = torch.topk(step, k=topk).indices.tolist()
        if blank_index not in top_indices:
            top_indices.append(blank_index)

        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = defaultdict(lambda: (-math.inf, -math.inf))
        for prefix, (p_blank, p_non_blank) in beams.items():
            blank_prob = float(step[blank_index].item())
            next_p_blank, next_p_non_blank = next_beams[prefix]
            next_beams[prefix] = (
                logsumexp_pair(next_p_blank, logsumexp_pair(p_blank + blank_prob, p_non_blank + blank_prob)),
                next_p_non_blank,
            )

            last_char = prefix[-1] if prefix else None
            for char_index in top_indices:
                if char_index == blank_index:
                    continue
                char_prob = float(step[char_index].item())
                extended_prefix = prefix + (char_index,)
                ext_blank, ext_non_blank = next_beams[extended_prefix]

                if char_index == last_char:
                    next_beams[extended_prefix] = (
                        ext_blank,
                        logsumexp_pair(ext_non_blank, p_blank + char_prob),
                    )
                    cur_blank, cur_non_blank = next_beams[prefix]
                    next_beams[prefix] = (
                        cur_blank,
                        logsumexp_pair(cur_non_blank, p_non_blank + char_prob),
                    )
                else:
                    next_beams[extended_prefix] = (
                        ext_blank,
                        logsumexp_pair(ext_non_blank, logsumexp_pair(p_blank + char_prob, p_non_blank + char_prob)),
                    )

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: logsumexp_pair(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix = max(beams.items(), key=lambda item: logsumexp_pair(item[1][0], item[1][1]))[0]
    return decode_indices(best_prefix)


@dataclass(frozen=True)
class SentenceRecord:
    session_name: str
    session_id: int
    sentence_index: int
    time_bins: int
    block_num: int
    transcript: str


class StanfordSentenceDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        session_names: Optional[Sequence[str]] = None,
        bin_size: int = 2,
        cache_size: int = 3,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.bin_size = max(1, int(bin_size))
        self.cache_size = max(1, int(cache_size))
        self.session_names = list(session_names) if session_names else self._discover_sessions()
        self.session_to_id = {name: idx for idx, name in enumerate(self.session_names)}
        self.records = self._load_records()
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def _discover_sessions(self) -> List[str]:
        sessions = []
        for name in sorted(os.listdir(self.data_root)):
            sentence_path = os.path.join(self.data_root, name, "sentences.mat")
            single_path = os.path.join(self.data_root, name, "singleLetters.mat")
            if os.path.exists(sentence_path) and os.path.exists(single_path):
                sessions.append(name)
        if not sessions:
            raise FileNotFoundError(f"No usable sessions found under {self.data_root}")
        return sessions

    def _load_records(self) -> List[SentenceRecord]:
        records: List[SentenceRecord] = []
        for session_name in self.session_names:
            sentence_path = os.path.join(self.data_root, session_name, "sentences.mat")
            meta = sio.loadmat(
                sentence_path,
                variable_names=[
                    "intendedText",
                    "sentencePrompt",
                    "numTimeBinsPerSentence",
                    "sentenceBlockNums",
                    "excludedSentences",
                ],
            )
            intended = meta.get("intendedText")
            prompts = meta.get("sentencePrompt")
            lengths = meta["numTimeBinsPerSentence"].reshape(-1)
            blocks = meta["sentenceBlockNums"].reshape(-1)
            excluded = meta.get("excludedSentences", np.zeros((len(lengths), 1), dtype=np.int32)).reshape(-1)

            for sentence_index in range(len(lengths)):
                if int(excluded[sentence_index]) != 0:
                    continue
                raw_text = ""
                if intended is not None and intended.size:
                    raw_text = unwrap_mat_string(intended.reshape(-1)[sentence_index])
                if not raw_text and prompts is not None and prompts.size:
                    raw_text = unwrap_mat_string(prompts.reshape(-1)[sentence_index])
                transcript = normalize_transcript(raw_text)
                if not transcript:
                    continue
                records.append(
                    SentenceRecord(
                        session_name=session_name,
                        session_id=self.session_to_id[session_name],
                        sentence_index=sentence_index,
                        time_bins=int(lengths[sentence_index]),
                        block_num=int(blocks[sentence_index]),
                        transcript=transcript,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_session_arrays(self, session_name: str) -> Dict[str, np.ndarray]:
        if session_name in self._cache:
            self._cache.move_to_end(session_name)
            return self._cache[session_name]

        sentence_path = os.path.join(self.data_root, session_name, "sentences.mat")
        single_path = os.path.join(self.data_root, session_name, "singleLetters.mat")
        sentence_dat = sio.loadmat(sentence_path, variable_names=["neuralActivityCube"])
        single_dat = sio.loadmat(single_path, variable_names=["meansPerBlock", "stdAcrossAllData", "blockList"])

        cache_entry = {
            "neuralActivityCube": sentence_dat["neuralActivityCube"],
            "meansPerBlock": single_dat["meansPerBlock"].astype(np.float32),
            "stdAcrossAllData": single_dat["stdAcrossAllData"].astype(np.float32).reshape(-1),
            "blockList": single_dat["blockList"].reshape(-1).astype(np.int32),
        }
        self._cache[session_name] = cache_entry
        self._cache.move_to_end(session_name)

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return cache_entry

    def _normalize_signal(self, signal: np.ndarray, block_num: int, session_arrays: Dict[str, np.ndarray]) -> np.ndarray:
        block_list = session_arrays["blockList"]
        block_idx = int(np.argmin(np.abs(block_list - int(block_num))))
        block_mean = session_arrays["meansPerBlock"][block_idx]
        block_std = session_arrays["stdAcrossAllData"]
        signal = signal.astype(np.float32) - block_mean[None, :]
        signal = signal / (block_std[None, :] + 1e-6)
        return signal

    def _bin_signal(self, signal: np.ndarray) -> np.ndarray:
        if self.bin_size <= 1:
            return signal
        total_steps = signal.shape[0]
        out_steps = math.ceil(total_steps / self.bin_size)
        padded_steps = out_steps * self.bin_size
        if padded_steps != total_steps:
            pad = np.repeat(signal[-1:, :], padded_steps - total_steps, axis=0)
            signal = np.concatenate([signal, pad], axis=0)
        signal = signal.reshape(out_steps, self.bin_size, signal.shape[1]).mean(axis=1)
        return signal

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        session_arrays = self._load_session_arrays(record.session_name)
        signal = session_arrays["neuralActivityCube"][record.sentence_index, : record.time_bins, :]
        signal = self._normalize_signal(signal, record.block_num, session_arrays)
        signal = self._bin_signal(signal)

        targets = torch.tensor(encode_transcript(record.transcript), dtype=torch.long)
        return {
            "inputs": torch.from_numpy(signal.astype(np.float32)),
            "input_length": signal.shape[0],
            "targets": targets,
            "target_length": len(targets),
            "session_id": record.session_id,
            "transcript": record.transcript,
            "display_text": render_transcript(record.transcript),
            "session_name": record.session_name,
            "sentence_index": record.sentence_index,
            "block_num": record.block_num,
        }


def make_block_split_indices(dataset: StanfordSentenceDataset, val_fraction: float = 0.2) -> Tuple[List[int], List[int]]:
    records_by_session: Dict[str, List[Tuple[int, SentenceRecord]]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        records_by_session[record.session_name].append((index, record))

    train_indices: List[int] = []
    val_indices: List[int] = []

    for session_name in dataset.session_names:
        session_records = records_by_session[session_name]
        unique_blocks = sorted({record.block_num for _, record in session_records})
        n_val_blocks = max(1, int(math.ceil(len(unique_blocks) * val_fraction)))
        val_blocks = set(unique_blocks[-n_val_blocks:])

        for index, record in session_records:
            if record.block_num in val_blocks:
                val_indices.append(index)
            else:
                train_indices.append(index)

    return train_indices, val_indices


def collate_sentence_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    batch_size = len(batch)
    input_lengths = torch.tensor([int(item["input_length"]) for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([int(item["target_length"]) for item in batch], dtype=torch.long)
    session_ids = torch.tensor([int(item["session_id"]) for item in batch], dtype=torch.long)

    max_input_length = int(input_lengths.max().item())
    feature_dim = int(batch[0]["inputs"].shape[1])
    padded_inputs = torch.zeros(batch_size, max_input_length, feature_dim, dtype=torch.float32)

    targets = []
    transcripts: List[str] = []
    display_texts: List[str] = []
    session_names: List[str] = []
    sentence_indices: List[int] = []
    block_nums: List[int] = []

    for row, item in enumerate(batch):
        current_inputs = item["inputs"]
        padded_inputs[row, : current_inputs.shape[0]] = current_inputs
        targets.append(item["targets"])
        transcripts.append(item["transcript"])
        display_texts.append(item["display_text"])
        session_names.append(item["session_name"])
        sentence_indices.append(int(item["sentence_index"]))
        block_nums.append(int(item["block_num"]))

    flat_targets = torch.cat(targets, dim=0)
    return {
        "inputs": padded_inputs,
        "input_lengths": input_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "session_ids": session_ids,
        "transcripts": transcripts,
        "display_texts": display_texts,
        "session_names": session_names,
        "sentence_indices": sentence_indices,
        "block_nums": block_nums,
    }


def mask_from_lengths(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    steps = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def apply_neural_noise(
    inputs: torch.Tensor,
    lengths: torch.Tensor,
    white_noise_sd: float,
    constant_offset_sd: float,
    random_walk_sd: float,
) -> torch.Tensor:
    if white_noise_sd <= 0 and constant_offset_sd <= 0 and random_walk_sd <= 0:
        return inputs

    batch_size, time_steps, channels = inputs.shape
    mask = mask_from_lengths(lengths, time_steps).unsqueeze(-1)
    noise = torch.zeros_like(inputs)

    if white_noise_sd > 0:
        noise = noise + torch.randn_like(inputs) * white_noise_sd
    if constant_offset_sd > 0:
        noise = noise + torch.randn(batch_size, 1, channels, device=inputs.device) * constant_offset_sd
    if random_walk_sd > 0:
        drift = torch.cumsum(torch.randn(batch_size, time_steps, channels, device=inputs.device) * random_walk_sd, dim=1)
        noise = noise + drift

    return inputs + noise * mask


class GaussianSmoothing1D(nn.Module):
    def __init__(self, channels: int, sigma: float = 2.0, truncate: float = 3.0) -> None:
        super().__init__()
        if sigma <= 0:
            kernel = torch.ones(1, 1, 1, dtype=torch.float32)
        else:
            radius = max(1, int(round(sigma * truncate)))
            positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-(positions ** 2) / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, -1)
        self.padding = kernel.shape[-1] // 2
        self.register_buffer("kernel", kernel.repeat(channels, 1, 1))
        self.channels = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.kernel.shape[-1] == 1:
            return inputs
        smoothed = F.conv1d(
            inputs.transpose(1, 2),
            self.kernel,
            padding=self.padding,
            groups=self.channels,
        )
        return smoothed.transpose(1, 2)


class BrainToTextV5(nn.Module):
    """
    A paper-inspired sentence decoder:
    1. session-specific affine calibration
    2. Gaussian temporal smoothing
    3. two unidirectional GRU layers
    4. slower top layer via temporal skipping
    5. CTC readout for full-sentence decoding
    """

    def __init__(
        self,
        input_size: int = 192,
        hidden_size: int = 512,
        vocab_size: int = len(VOCAB),
        num_sessions: int = 11,
        smoothing_sigma: float = 2.0,
        skip_len: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        eye = torch.eye(input_size, dtype=torch.float32).unsqueeze(0).repeat(num_sessions, 1, 1)
        self.session_weight = nn.Parameter(eye)
        self.session_bias = nn.Parameter(torch.zeros(num_sessions, input_size))
        self.smoother = GaussianSmoothing1D(input_size, sigma=smoothing_sigma)
        self.dropout = nn.Dropout(dropout)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_size, vocab_size + 1)
        self.skip_len = max(1, int(skip_len))

    def _apply_session_affine(self, inputs: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
        weights = self.session_weight[session_ids]
        bias = self.session_bias[session_ids].unsqueeze(1)
        return torch.bmm(inputs, weights.transpose(1, 2)) + bias

    def _subsample(self, sequence: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        subsampled = sequence[:, :: self.skip_len, :]
        new_lengths = torch.div(lengths + (self.skip_len - 1), self.skip_len, rounding_mode="floor")
        return subsampled, new_lengths.clamp_min(1)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        session_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._apply_session_affine(inputs, session_ids)
        x = self.smoother(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out_1, _ = self.gru1(packed)
        out_1, _ = pad_packed_sequence(packed_out_1, batch_first=True)
        out_1 = self.dropout(out_1)

        out_1, reduced_lengths = self._subsample(out_1, lengths)
        packed_2 = pack_padded_sequence(out_1, reduced_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out_2, _ = self.gru2(packed_2)
        out_2, _ = pad_packed_sequence(packed_out_2, batch_first=True)
        out_2 = self.dropout(out_2)

        logits = self.classifier(out_2)
        return logits, reduced_lengths


def decode_batch(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    beam_width: int = 1,
    blank_index: int = BLANK_INDEX,
) -> List[str]:
    log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
    outputs: List[str] = []
    for row in range(log_probs.shape[0]):
        current = log_probs[row, : int(lengths[row].item())]
        if beam_width > 1:
            outputs.append(ctc_prefix_beam_search(current, beam_width=beam_width, blank_index=blank_index))
        else:
            outputs.append(ctc_greedy_decode(current, blank_index=blank_index))
    return outputs


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    beam_width: int = 1,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    ctc_loss = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    references: List[str] = []
    hypotheses: List[str] = []
    examples: List[Tuple[str, str]] = []
    total_loss = 0.0
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = batch["inputs"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            session_ids = batch["session_ids"].to(device)

            logits, logit_lengths = model(inputs, input_lengths, session_ids)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = ctc_loss(log_probs, targets, logit_lengths.cpu(), target_lengths.cpu())

            decoded = decode_batch(logits, logit_lengths, beam_width=beam_width)
            references.extend(batch["transcripts"])
            hypotheses.extend(decoded)
            total_loss += float(loss.item())
            n_batches += 1

            if len(examples) < 5:
                for reference, hypothesis in zip(batch["transcripts"], decoded):
                    if len(examples) >= 5:
                        break
                    examples.append((render_transcript(reference), render_transcript(hypothesis)))

    metrics = compute_error_rates(references, hypotheses)
    metrics["loss"] = total_loss / max(1, n_batches)
    metrics["examples"] = examples
    return metrics


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[BrainToTextV5, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = BrainToTextV5(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def build_dataset_from_checkpoint(
    checkpoint: Dict[str, object],
    data_root: str,
) -> Tuple[StanfordSentenceDataset, Subset]:
    dataset = StanfordSentenceDataset(
        data_root=data_root,
        session_names=checkpoint["session_names"],
        bin_size=int(checkpoint["data_config"]["bin_size"]),
    )
    split_indices = checkpoint.get("val_indices")
    if split_indices is None:
        _, split_indices = make_block_split_indices(dataset, val_fraction=float(checkpoint["data_config"]["val_fraction"]))
    return dataset, Subset(dataset, split_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V5 sentence-level brain-to-text inference.")
    parser.add_argument("--checkpoint", type=str, default="brain_to_text_v5_best.pt")
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "Datasets"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--beam_width", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    _, val_dataset = build_dataset_from_checkpoint(checkpoint, args.data_root)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sentence_batch)

    metrics = evaluate_model(model, val_loader, device, beam_width=args.beam_width, max_batches=args.max_batches)
    print(
        f"Loss={metrics['loss']:.4f} | CER={metrics['cer'] * 100:.2f}% | "
        f"WER={metrics['wer'] * 100:.2f}%"
    )
    print("-" * 60)
    for reference, hypothesis in metrics["examples"]:
        print(f"REF: {reference}")
        print(f"HYP: {hypothesis}")
        print("-" * 60)


if __name__ == "__main__":
    main()
