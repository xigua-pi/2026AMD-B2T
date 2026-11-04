from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

from brain_to_text_main_v5 import (
    StanfordSentenceDataset,
    compute_error_rates,
    make_block_split_indices,
    normalize_transcript,
    render_transcript,
    unwrap_mat_string,
)


def get_handwriting_character_definitions() -> Dict[str, object]:
    char_def: Dict[str, object] = {}
    char_def["charList"] = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "greaterThan", "comma",
        "apostrophe", "tilde", "questionMark",
    ]
    char_def["charListAbbr"] = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", ">", ",", "'", "~", "?",
    ]
    char_def["charLen"] = np.array(
        [99, 91, 70, 104, 98, 125, 110, 104, 79, 92, 127, 68, 132, 90, 84, 113, 104, 74,
         86, 110, 86, 83, 110, 103, 115, 100, 82, 77, 116, 71, 110],
        dtype=np.int32,
    )
    char_def["penStart"] = [
        0.25, 1.0, 0.5, 0.5, 0.25, 1.0, 0.25, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.25,
        0.5, 0.25, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 1.0, 0.5, 1.0,
    ]
    char_def["strToCharIdx"] = {
        char_abbr: idx for idx, char_abbr in enumerate(char_def["charListAbbr"])
    }
    char_def["abbrToFull"] = {
        char_abbr: char_full
        for char_full, char_abbr in zip(char_def["charList"], char_def["charListAbbr"])
    }
    return char_def


CHAR_DEF = get_handwriting_character_definitions()
N_CHAR_CLASSES = len(CHAR_DEF["charList"])


def resample_time_major(data: np.ndarray, new_len: int) -> np.ndarray:
    if data.shape[0] == new_len:
        return data.astype(np.float32)
    if data.shape[0] <= 1:
        return np.repeat(data[:1], new_len, axis=0).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, data.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, new_len, dtype=np.float32)
    out = np.zeros((new_len, data.shape[1]), dtype=np.float32)
    for channel in range(data.shape[1]):
        out[:, channel] = np.interp(new_x, old_x, data[:, channel])
    return out


def bin_tensor_time_major(data: np.ndarray, bin_size: int) -> np.ndarray:
    if bin_size <= 1:
        return data.astype(np.float32)
    n_steps = int(math.ceil(data.shape[0] / float(bin_size)))
    padded = n_steps * bin_size
    if padded != data.shape[0]:
        pad = np.repeat(data[-1:, :], padded - data.shape[0], axis=0)
        data = np.concatenate([data, pad], axis=0)
    return data.reshape(n_steps, bin_size, data.shape[1]).mean(axis=1).astype(np.float32)


def bin_vector(data: np.ndarray, bin_size: int) -> np.ndarray:
    if bin_size <= 1:
        return data.astype(np.float32)
    n_steps = int(math.ceil(data.shape[0] / float(bin_size)))
    padded = n_steps * bin_size
    if padded != data.shape[0]:
        pad_value = data[-1] if data.size else 0.0
        pad = np.full((padded - data.shape[0],), pad_value, dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)
    return data.reshape(n_steps, bin_size).mean(axis=1).astype(np.float32)


def low_rank_denoise(template: np.ndarray, n_components: int = 10) -> np.ndarray:
    if template.shape[0] < 2 or template.shape[1] < 2:
        return template.astype(np.float32)
    centered = template - np.mean(template, axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(n_components, s.shape[0])
    reconstructed = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    reconstructed = reconstructed + np.mean(template, axis=0, keepdims=True)
    return reconstructed.astype(np.float32)


def pick_word_for_sentence(
    word_list: Sequence[str],
    rare_letter_increase: bool = False,
    rare_word_indices: Sequence[int] | None = None,
) -> List[str]:
    rare_word_indices = list(rare_word_indices or [])
    if len(word_list) == 0:
        return [">"]

    if np.random.rand() < 0.2:
        word_index = np.random.randint(min(20, len(word_list)))
    elif rare_letter_increase and len(rare_word_indices) > 0 and np.random.rand() < 0.2:
        word_index = rare_word_indices[np.random.randint(len(rare_word_indices))]
    else:
        word_index = np.random.randint(len(word_list))

    next_word = list(word_list[word_index])
    if np.random.rand() < 0.03 and len(next_word) > 3:
        next_word.insert(len(next_word) - 1, "'")

    put_comma = np.random.rand() < 0.07
    put_period = (not put_comma) and (np.random.rand() < 0.05)
    put_question = (not put_comma) and (not put_period) and (np.random.rand() < 0.05)

    if put_comma:
        next_word.extend(",")
    if put_period:
        next_word.extend("~")
    if put_question:
        next_word.extend("?")
    if not put_period and not put_question:
        next_word.extend(">")
    return next_word


@dataclass
class SentenceLabelRecord:
    session_name: str
    session_id: int
    sentence_index: int
    block_num: int
    time_bins: int
    transcript: str
    split: str
    letter_starts: np.ndarray
    letter_durations: np.ndarray
    blank_windows: List[Tuple[int, int]]

    def to_serializable(self) -> Dict[str, object]:
        return {
            "session_name": self.session_name,
            "session_id": self.session_id,
            "sentence_index": self.sentence_index,
            "block_num": self.block_num,
            "time_bins": self.time_bins,
            "transcript": self.transcript,
            "split": self.split,
            "letter_starts": self.letter_starts.astype(np.float32).tolist(),
            "letter_durations": self.letter_durations.astype(np.float32).tolist(),
            "blank_windows": [[int(s), int(e)] for s, e in self.blank_windows],
        }

    @staticmethod
    def from_serializable(payload: Dict[str, object]) -> "SentenceLabelRecord":
        return SentenceLabelRecord(
            session_name=str(payload["session_name"]),
            session_id=int(payload["session_id"]),
            sentence_index=int(payload["sentence_index"]),
            block_num=int(payload["block_num"]),
            time_bins=int(payload["time_bins"]),
            transcript=str(payload["transcript"]),
            split=str(payload["split"]),
            letter_starts=np.asarray(payload["letter_starts"], dtype=np.float32),
            letter_durations=np.asarray(payload["letter_durations"], dtype=np.float32),
            blank_windows=[(int(pair[0]), int(pair[1])) for pair in payload["blank_windows"]],
        )


class RawSessionDataManager:
    def __init__(self, data_root: str, cache_size: int = 1) -> None:
        self.data_root = data_root
        self.cache_size = max(1, cache_size)
        self._sentence_cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
        self._single_cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def _trim_cache(self, cache: OrderedDict[str, Dict[str, np.ndarray]]) -> None:
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def _load_sentence_file(self, session_name: str) -> Dict[str, np.ndarray]:
        if session_name in self._sentence_cache:
            self._sentence_cache.move_to_end(session_name)
            return self._sentence_cache[session_name]

        path = os.path.join(self.data_root, session_name, "sentences.mat")
        payload = sio.loadmat(
            path,
            variable_names=["neuralActivityCube", "numTimeBinsPerSentence", "sentenceBlockNums"],
        )
        self._sentence_cache[session_name] = payload
        self._sentence_cache.move_to_end(session_name)
        self._trim_cache(self._sentence_cache)
        return payload

    def _load_single_file(self, session_name: str) -> Dict[str, np.ndarray]:
        if session_name in self._single_cache:
            self._single_cache.move_to_end(session_name)
            return self._single_cache[session_name]

        path = os.path.join(self.data_root, session_name, "singleLetters.mat")
        payload = sio.loadmat(path)
        self._single_cache[session_name] = payload
        self._single_cache.move_to_end(session_name)
        self._trim_cache(self._single_cache)
        return payload

    def get_sentence_signal(
        self,
        session_name: str,
        sentence_index: int,
        time_bins: int,
        block_num: int,
        smooth_sigma: float = 4.0,
    ) -> np.ndarray:
        sentence_dat = self._load_sentence_file(session_name)
        single_dat = self._load_single_file(session_name)

        signal = sentence_dat["neuralActivityCube"][sentence_index, :time_bins, :].astype(np.float32)
        block_list = single_dat["blockList"].reshape(-1).astype(np.int32)
        block_idx = int(np.argmin(np.abs(block_list - int(block_num))))
        signal = signal - single_dat["meansPerBlock"][block_idx].astype(np.float32)[None, :]
        signal = signal / (single_dat["stdAcrossAllData"].reshape(-1).astype(np.float32)[None, :] + 1e-6)
        if smooth_sigma > 0:
            signal = gaussian_filter1d(signal, smooth_sigma, axis=0).astype(np.float32)
        return signal

    def get_single_letter_trials(
        self,
        session_name: str,
        char_name: str,
        smooth_sigma: float = 4.0,
    ) -> List[np.ndarray]:
        single_dat = self._load_single_file(session_name)
        key = f"neuralActivityCube_{char_name}"
        if key not in single_dat:
            return []

        cube = single_dat[key].astype(np.float32)
        means = single_dat["meansPerBlock"].astype(np.float32)
        std = single_dat["stdAcrossAllData"].reshape(-1).astype(np.float32)
        block_list = single_dat["blockList"].reshape(-1).astype(np.int32)

        char_cues = [str(x[0]) if isinstance(x, np.ndarray) else str(x) for x in single_dat["characterCues"].reshape(-1)]
        trial_indices = [idx for idx, cue in enumerate(char_cues) if cue == char_name]
        if len(trial_indices) == cube.shape[0]:
            go_bins = single_dat["goPeriodOnsetTimeBin"].reshape(-1).astype(np.int32)[trial_indices]
            time_series_blocks = single_dat["blockNumsTimeSeries"].reshape(-1).astype(np.int32)
            trial_blocks = time_series_blocks[go_bins]
        else:
            trial_blocks = np.full((cube.shape[0],), block_list[0], dtype=np.int32)

        trials: List[np.ndarray] = []
        for trial_idx in range(cube.shape[0]):
            block_idx = int(np.argmin(np.abs(block_list - int(trial_blocks[trial_idx]))))
            trial = cube[trial_idx] - means[block_idx][None, :]
            trial = trial / (std[None, :] + 1e-6)
            if smooth_sigma > 0:
                trial = gaussian_filter1d(trial, smooth_sigma, axis=0).astype(np.float32)
            trials.append(trial.astype(np.float32))
        return trials


def initialize_templates_from_single_letters(
    manager: RawSessionDataManager,
    session_name: str,
    char_def: Dict[str, object],
) -> Dict[str, np.ndarray]:
    templates: Dict[str, np.ndarray] = {}
    for char_name, char_abbr, char_len in zip(
        char_def["charList"],
        char_def["charListAbbr"],
        char_def["charLen"],
    ):
        trials = manager.get_single_letter_trials(session_name, char_name)
        warped_trials: List[np.ndarray] = []
        for trial in trials:
            start = min(59, max(0, trial.shape[0] - 5))
            snippet = trial[start:, :]
            if snippet.shape[0] < 10:
                snippet = trial
            warped_trials.append(resample_time_major(snippet, int(char_len) + 1))

        if len(warped_trials) == 0:
            templates[char_abbr] = np.zeros((int(char_len) + 1, 192), dtype=np.float32)
            continue

        mean_template = np.mean(np.stack(warped_trials, axis=0), axis=0)
        templates[char_abbr] = low_rank_denoise(mean_template, n_components=10)
    return templates


def make_forced_alignment_hmm(
    templates: Dict[str, np.ndarray],
    sentence: str,
    hmm_bin_size: int,
    blank_prob: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stay_prob = 0.20
    skip_prob = 0.20

    n_states = 0
    letter_start_idx = np.zeros(len(sentence), dtype=np.int32)
    for idx, char in enumerate(sentence):
        letter_start_idx[idx] = n_states
        n_states += int(np.floor(templates[char].shape[0] / hmm_bin_size)) + 1

    n_states = int(n_states + 1)
    letter_start_idx = letter_start_idx + 1

    a_hmm = np.zeros((n_states, n_states), dtype=np.float32)
    b_hmm = np.zeros((n_states, templates[sentence[0]].shape[1]), dtype=np.float32)
    state_labels = np.empty((n_states, 1), dtype=object)
    state_labels_seq = np.zeros((n_states, 1), dtype=np.int32)

    for char_pos, char in enumerate(sentence):
        n_bins = int(np.floor(templates[char].shape[0] / hmm_bin_size))
        idx_in_template = np.arange(0, hmm_bin_size, dtype=np.int32)
        current_state = int(letter_start_idx[char_pos])

        for bin_index in range(n_bins):
            mean_rates = np.mean(templates[char][idx_in_template, :], axis=0)
            b_hmm[current_state, :] = mean_rates
            state_labels[current_state] = char
            state_labels_seq[current_state] = char_pos

            a_hmm[current_state, current_state] = stay_prob
            if bin_index < (n_bins - 1):
                if bin_index == (n_bins - 2):
                    a_hmm[current_state, current_state + 1] = 1.0 - stay_prob
                else:
                    a_hmm[current_state, current_state + 1] = 1.0 - stay_prob - skip_prob
                    a_hmm[current_state, current_state + 2] = skip_prob
            else:
                a_hmm[current_state, current_state + 1] = (1.0 - stay_prob) * blank_prob
                a_hmm[current_state + 1, current_state + 1] = 0.5
                state_labels[current_state + 1] = "&"
                state_labels_seq[current_state + 1] = -1

                if char_pos < (len(sentence) - 1):
                    a_hmm[current_state, letter_start_idx[char_pos + 1]] = (1.0 - stay_prob) * (1.0 - blank_prob)
                    a_hmm[current_state + 1, letter_start_idx[char_pos + 1]] = 0.5
                else:
                    a_hmm[current_state, current_state + 1] = 1.0 - stay_prob
                    a_hmm[current_state + 1, current_state + 1] = 1.0

            current_state += 1
            idx_in_template += hmm_bin_size

    state_labels[0] = "&"
    state_labels_seq[0] = -1
    a_hmm[0, 0] = 0.5
    a_hmm[0, 1] = 0.5

    letter_states = state_labels_seq[:, 0] >= 0
    blank_states = state_labels_seq[:, 0] == -1
    blank_rates = np.mean(b_hmm[letter_states, :], axis=0, keepdims=True)
    b_hmm[blank_states, :] = blank_rates

    diag_variance = np.ones((1, b_hmm.shape[1]), dtype=np.float32)
    return a_hmm, b_hmm, diag_variance, state_labels, state_labels_seq


def make_time_window_mask(state_labels_seq: np.ndarray, n_time_steps: int) -> np.ndarray:
    n_states = state_labels_seq.shape[0]
    n_char = int(np.max(state_labels_seq))
    time_window_mask = np.zeros((n_time_steps, n_states), dtype=np.float32)

    if n_char == 1:
        win_len = 1.0
    elif n_char <= 10:
        win_len = 0.50
    else:
        win_len = 0.30

    for state_index in range(n_states):
        sl = int(state_labels_seq[state_index, 0])
        if sl == -1:
            sl = int(state_labels_seq[max(0, state_index - 1), 0])
        elif sl == -2:
            sl = int(state_labels_seq[max(0, state_index - 2), 0])

        time_fraction = 0.0 if n_char <= 0 else sl / float(n_char)
        t_idx = np.arange(
            round(time_fraction * n_time_steps - win_len * n_time_steps),
            round(time_fraction * n_time_steps + win_len * n_time_steps),
            dtype=np.int32,
        )
        t_idx = t_idx[(t_idx >= 0) & (t_idx < n_time_steps)]
        time_window_mask[t_idx, state_index] = 1.0

    if n_time_steps > 0:
        time_window_mask[-1, 0:-2] = 0.0
    return time_window_mask


def hmm_viterbi(
    obs: np.ndarray,
    a_hmm: np.ndarray,
    b_hmm: np.ndarray,
    diag_variance: np.ndarray,
    time_window_mask: np.ndarray,
    start_prob: np.ndarray,
) -> np.ndarray:
    num_states = a_hmm.shape[0]
    length = obs.shape[0]

    with np.errstate(divide="ignore"):
        log_tr = np.log(a_hmm)
        v = np.log(start_prob[:, np.newaxis])

    ptr = np.zeros((num_states, length), dtype=np.int32)
    for step in range(length):
        squared_diff = -np.square(b_hmm - obs[step : step + 1, :]) / (2.0 * diag_variance)
        gaussian_emission = np.sum(squared_diff, axis=1, keepdims=True)

        tmp_v = v + log_tr
        max_idx = np.argmax(tmp_v, axis=0)
        max_val = np.take_along_axis(tmp_v, np.expand_dims(max_idx, axis=0), axis=0)
        with np.errstate(divide="ignore"):
            v = gaussian_emission + np.transpose(max_val) + np.log(time_window_mask[step, :, np.newaxis])
        ptr[:, step] = max_idx

    final_state = int(np.argmax(v))
    viterbi_states = np.zeros((length,), dtype=np.int32)
    viterbi_states[-1] = final_state
    for step in range(length - 2, 0, -1):
        viterbi_states[step] = ptr[viterbi_states[step + 1], step + 1]
    return viterbi_states


def refine_character_start_times(
    obs: np.ndarray,
    sentence: str,
    templates: Dict[str, np.ndarray],
    letter_starts: np.ndarray,
    letter_stretches: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    letter_starts = letter_starts.astype(np.float32).copy()
    letter_stretches = letter_stretches.astype(np.float32).copy()

    for char_idx in range(len(sentence)):
        possible_start = np.arange(letter_starts[char_idx] - 50, letter_starts[char_idx] + 55, 5).astype(np.int32)
        possible_start = possible_start[possible_start >= 0]
        possible_stretch = np.linspace(0.4, 1.5, 15, dtype=np.float32)

        if char_idx > 0:
            possible_start = possible_start[possible_start >= int(letter_starts[char_idx - 1] + 20)]
        if possible_start.size == 0:
            continue

        template = templates[sentence[char_idx]]
        corr_heatmap = np.full((len(possible_stretch), len(possible_start)), -np.inf, dtype=np.float32)

        for stretch_count, stretch in enumerate(possible_stretch):
            stretched = resample_time_major(template, max(2, int(round(template.shape[0] * float(stretch)))))
            for start_count, start_idx in enumerate(possible_start):
                if char_idx > 0:
                    prev_end = letter_starts[char_idx - 1] + letter_stretches[char_idx - 1] * templates[sentence[char_idx - 1]].shape[0]
                    if start_idx < (prev_end - 10):
                        continue

                if char_idx < (len(sentence) - 1):
                    this_end = start_idx + stretched.shape[0]
                    if this_end > letter_starts[char_idx + 1] + 10:
                        continue

                step_idx = np.arange(start_idx, start_idx + stretched.shape[0], dtype=np.int32)
                if len(step_idx) == 0 or step_idx[-1] >= obs.shape[0]:
                    continue

                ms_dat = obs[step_idx, :] - np.mean(obs[step_idx, :], axis=0, keepdims=True)
                ms_template = stretched - np.mean(stretched, axis=0, keepdims=True)
                norm_dat = np.sqrt(np.sum(np.square(ms_dat), axis=0))
                norm_template = np.sqrt(np.sum(np.square(ms_template), axis=0))
                with np.errstate(invalid="ignore"):
                    corr_coeff = np.sum(ms_dat * ms_template, axis=0) / (norm_dat * norm_template)
                corr_heatmap[stretch_count, start_count] = np.nanmean(corr_coeff)

        max_idx = np.unravel_index(int(np.argmax(corr_heatmap)), corr_heatmap.shape)
        letter_stretches[char_idx] = float(possible_stretch[max_idx[0]])
        letter_starts[char_idx] = float(possible_start[max_idx[1]])

    return letter_starts, letter_stretches


def forced_alignment_labeling(
    obs: np.ndarray,
    sentence: str,
    templates: Dict[str, np.ndarray],
    hmm_bin_size: int = 5,
    blank_prob: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    a_hmm, b_hmm, diag_variance, _state_labels, state_labels_seq = make_forced_alignment_hmm(
        templates=templates,
        sentence=sentence,
        hmm_bin_size=hmm_bin_size,
        blank_prob=blank_prob,
    )

    n_bins = int(np.floor(obs.shape[0] / float(hmm_bin_size)))
    if n_bins <= 1:
        return np.array([0.0], dtype=np.float32), np.array([float(obs.shape[0])], dtype=np.float32), []

    obs_binned = np.zeros((n_bins, obs.shape[1]), dtype=np.float32)
    idx = np.arange(0, hmm_bin_size, dtype=np.int32)
    for bin_index in range(n_bins):
        obs_binned[bin_index, :] = np.mean(obs[idx, :], axis=0)
        idx += hmm_bin_size

    time_window_mask = make_time_window_mask(state_labels_seq, obs_binned.shape[0])
    start_prob = np.zeros((a_hmm.shape[0],), dtype=np.float32)
    start_prob[0] = blank_prob
    start_prob[1] = 1.0 - start_prob[0]
    viterbi_states = hmm_viterbi(obs_binned, a_hmm, b_hmm, diag_variance, time_window_mask, start_prob)

    labeled_states = state_labels_seq[viterbi_states]
    letter_starts = np.zeros((len(sentence),), dtype=np.float32)
    letter_stretches = np.zeros((len(sentence),), dtype=np.float32)
    for char_pos in range(len(sentence)):
        current = np.argwhere(labeled_states[:, 0] == char_pos)
        if len(current) == 0:
            fallback = 0 if char_pos == 0 else int(letter_starts[char_pos - 1] + 20)
            letter_starts[char_pos] = fallback
            letter_stretches[char_pos] = 1.0
            continue
        letter_starts[char_pos] = float((current[0, 0] + 1) * hmm_bin_size - 1)
        template_len = max(1, templates[sentence[char_pos]].shape[0])
        letter_stretches[char_pos] = float((len(current) * hmm_bin_size) / template_len)

    letter_starts, letter_stretches = refine_character_start_times(
        obs=obs,
        sentence=sentence,
        templates=templates,
        letter_starts=letter_starts,
        letter_stretches=letter_stretches,
    )

    letter_durations = np.zeros_like(letter_starts)
    for char_pos in range(len(sentence)):
        letter_durations[char_pos] = float(letter_stretches[char_pos] * templates[sentence[char_pos]].shape[0])

    padded_states = np.concatenate([np.array([0]), labeled_states[:, 0], np.array([0])], axis=0)
    blank_start = np.argwhere((padded_states[1:] == -1) & (padded_states[:-1] != -1))
    blank_end = np.argwhere((padded_states[:-1] == -1) & (padded_states[1:] != -1)) - 1
    blank_windows: List[Tuple[int, int]] = []
    for blank_idx in range(len(blank_start)):
        start_step = int((blank_start[blank_idx, 0] + 1) * hmm_bin_size - 1)
        end_step = int((blank_end[blank_idx, 0] + 2) * hmm_bin_size - 1)
        if end_step > start_step:
            blank_windows.append((start_step, end_step))

    return letter_starts.astype(np.float32), letter_durations.astype(np.float32), blank_windows


def update_templates_from_labels(
    manager: RawSessionDataManager,
    train_records: Sequence[SentenceLabelRecord],
    templates: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    new_templates = {key: np.zeros_like(value) for key, value in templates.items()}
    char_count = {key: 0 for key in templates.keys()}

    for record in train_records:
        signal = manager.get_sentence_signal(
            session_name=record.session_name,
            sentence_index=record.sentence_index,
            time_bins=record.time_bins,
            block_num=record.block_num,
        )
        for char_pos, char_abbr in enumerate(record.transcript):
            start = int(round(record.letter_starts[char_pos]))
            duration = max(5, int(round(record.letter_durations[char_pos])))
            end = min(signal.shape[0], start + duration)
            if end <= start:
                continue
            snippet = signal[start:end, :]
            if snippet.shape[0] < 3:
                continue
            resampled = resample_time_major(snippet, templates[char_abbr].shape[0])
            new_templates[char_abbr] += resampled
            char_count[char_abbr] += 1

    for char_abbr in templates.keys():
        if char_count[char_abbr] >= 10:
            new_templates[char_abbr] /= float(char_count[char_abbr])
            new_templates[char_abbr] = low_rank_denoise(new_templates[char_abbr], n_components=10)
        else:
            new_templates[char_abbr] = templates[char_abbr].copy()
    return new_templates


def _has_complete_official_root(root_path: str) -> bool:
    return os.path.isdir(os.path.join(root_path, "Datasets")) and os.path.isdir(
        os.path.join(root_path, "RNNTrainingSteps")
    )


def _find_complete_official_root(search_root: str, max_depth: int = 6) -> Optional[str]:
    if not os.path.isdir(search_root):
        return None

    search_root = os.path.abspath(search_root)
    if _has_complete_official_root(search_root):
        return search_root

    base_depth = search_root.rstrip("\\/").count(os.sep)
    for current_root, dirnames, _file_names in os.walk(search_root):
        current_depth = current_root.rstrip("\\/").count(os.sep) - base_depth
        if current_depth > max_depth:
            dirnames[:] = []
            continue
        if "Datasets" in dirnames and "RNNTrainingSteps" in dirnames:
            return current_root
    return None


def resolve_handwriting_data_roots(
    data_root: str,
    training_steps_root: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    data_root = os.path.abspath(data_root)

    if training_steps_root:
        resolved_steps = os.path.abspath(training_steps_root)
        resolved_data = (
            os.path.join(data_root, "Datasets")
            if _has_complete_official_root(data_root)
            else data_root
        )
        return resolved_data, resolved_steps

    if _has_complete_official_root(data_root):
        return os.path.join(data_root, "Datasets"), os.path.join(data_root, "RNNTrainingSteps")

    if os.path.basename(data_root).lower() == "datasets":
        sibling_steps = os.path.join(os.path.dirname(data_root), "RNNTrainingSteps")
        if os.path.isdir(sibling_steps):
            return data_root, sibling_steps

        discovered_root = _find_complete_official_root(os.path.dirname(data_root))
        if discovered_root:
            return os.path.join(discovered_root, "Datasets"), os.path.join(discovered_root, "RNNTrainingSteps")

    discovered_root = _find_complete_official_root(data_root)
    if discovered_root:
        return os.path.join(discovered_root, "Datasets"), os.path.join(discovered_root, "RNNTrainingSteps")

    return data_root, None


def _parse_blank_windows(blank_entry: object) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    if blank_entry is None:
        return windows

    if isinstance(blank_entry, np.ndarray) and blank_entry.dtype == object:
        iterable = blank_entry.reshape(-1).tolist()
    elif isinstance(blank_entry, (list, tuple)):
        iterable = list(blank_entry)
    else:
        iterable = [blank_entry]

    for item in iterable:
        idx = np.asarray(item).reshape(-1)
        if idx.size == 0:
            continue
        start = int(np.min(idx))
        end = int(np.max(idx)) + 1
        if end > start:
            windows.append((start, end))
    return windows


def _trim_step2_sequence(values: np.ndarray, n_chars: int) -> np.ndarray:
    if n_chars <= 0:
        return np.zeros((0,), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values[:n_chars].astype(np.float32)


class OfficialStepLabelManager:
    def __init__(self, training_steps_root: str, cv_part: str = "HeldOutBlocks", cache_size: int = 1) -> None:
        self.training_steps_root = training_steps_root
        self.cv_part = cv_part
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def _trim_cache(self) -> None:
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _load_label_file(self, session_name: str) -> Dict[str, np.ndarray]:
        if session_name in self._cache:
            self._cache.move_to_end(session_name)
            return self._cache[session_name]

        path = os.path.join(
            self.training_steps_root,
            "Step2_HMMLabels",
            self.cv_part,
            f"{session_name}_timeSeriesLabels.mat",
        )
        payload = sio.loadmat(
            path,
            variable_names=[
                "letterStarts",
                "letterDurations",
                "charStartTarget",
                "charProbTarget",
                "ignoreErrorHere",
                "blankWindows",
                "timeBinsPerSentence",
            ],
        )
        self._cache[session_name] = payload
        self._cache.move_to_end(session_name)
        self._trim_cache()
        return payload

    def get_sentence_targets(
        self,
        session_name: str,
        sentence_index: int,
        time_bins: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        payload = self._load_label_file(session_name)
        n_steps = int(time_bins)
        char_prob = np.asarray(payload["charProbTarget"][sentence_index, :n_steps, :], dtype=np.float32)
        char_start = np.asarray(payload["charStartTarget"][sentence_index, :n_steps], dtype=np.float32)
        ignore_error = np.asarray(payload["ignoreErrorHere"][sentence_index, :n_steps], dtype=np.float32)
        return char_prob, char_start, ignore_error


class WarpedCubeManager:
    def __init__(self, training_steps_root: str, cache_size: int = 1) -> None:
        self.training_steps_root = training_steps_root
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def _trim_cache(self) -> None:
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _load_file(self, session_name: str) -> Dict[str, np.ndarray]:
        if session_name in self._cache:
            self._cache.move_to_end(session_name)
            return self._cache[session_name]

        path = os.path.join(
            self.training_steps_root,
            "Step1_TimeWarping",
            f"{session_name}_warpedCubes.mat",
        )
        payload = sio.loadmat(path)
        self._cache[session_name] = payload
        self._cache.move_to_end(session_name)
        self._trim_cache()
        return payload

    def get_char_trials(self, session_name: str, char_name: str) -> List[np.ndarray]:
        payload = self._load_file(session_name)
        if char_name not in payload:
            return []
        cube = np.asarray(payload[char_name], dtype=np.float32)
        if cube.ndim != 3:
            return []
        trials: List[np.ndarray] = []
        for idx in range(cube.shape[0]):
            trial = np.nan_to_num(cube[idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if trial.shape[0] >= 5:
                trials.append(trial)
        return trials


def prepare_paper_v5_label_cache_official(
    data_root: str,
    training_steps_root: str,
    cache_path: str,
    cv_part: str = "HeldOutBlocks",
    force_rebuild: bool = False,
    max_records_per_session: Optional[int] = None,
) -> Dict[str, object]:
    if os.path.exists(cache_path) and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        payload["records"] = [SentenceLabelRecord.from_serializable(item) for item in payload["records"]]
        if payload.get("label_mode") == "official" and payload.get("cv_part") == cv_part:
            return payload

    label_dir = os.path.join(training_steps_root, "Step2_HMMLabels", cv_part)
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Official Step2 labels not found: {label_dir}")

    partition_path = os.path.join(training_steps_root, f"trainTestPartitions_{cv_part}.mat")
    partitions = sio.loadmat(partition_path)

    session_names = sorted(
        file_name.replace("_timeSeriesLabels.mat", "")
        for file_name in os.listdir(label_dir)
        if file_name.endswith("_timeSeriesLabels.mat")
    )

    all_records: List[SentenceLabelRecord] = []
    for session_id, session_name in enumerate(session_names):
        sentence_path = os.path.join(data_root, session_name, "sentences.mat")
        if not os.path.exists(sentence_path):
            continue

        sentence_meta = sio.loadmat(
            sentence_path,
            variable_names=[
                "intendedText",
                "sentencePrompt",
                "numTimeBinsPerSentence",
                "sentenceBlockNums",
                "excludedSentences",
            ],
        )
        label_payload = sio.loadmat(
            os.path.join(label_dir, f"{session_name}_timeSeriesLabels.mat"),
            variable_names=["letterStarts", "letterDurations", "blankWindows", "timeBinsPerSentence"],
        )

        intended = sentence_meta.get("intendedText")
        prompts = sentence_meta.get("sentencePrompt")
        lengths = sentence_meta["numTimeBinsPerSentence"].reshape(-1)
        blocks = sentence_meta["sentenceBlockNums"].reshape(-1)
        excluded = sentence_meta.get("excludedSentences", np.zeros((len(lengths), 1), dtype=np.int32)).reshape(-1)
        label_lengths = label_payload["timeBinsPerSentence"].reshape(-1)

        train_indices = set(int(x) for x in partitions.get(f"{session_name}_train", np.zeros((0, 0), dtype=np.int32)).reshape(-1))
        test_indices = set(int(x) for x in partitions.get(f"{session_name}_test", np.zeros((0, 0), dtype=np.int32)).reshape(-1))

        session_records: List[SentenceLabelRecord] = []
        n_sentences = min(len(lengths), label_payload["letterStarts"].shape[0], len(label_lengths))
        for sentence_index in range(n_sentences):
            if int(excluded[sentence_index]) != 0:
                continue

            split = None
            if sentence_index in train_indices:
                split = "train"
            elif sentence_index in test_indices:
                split = "val"
            else:
                continue

            if intended is not None:
                raw_text = unwrap_mat_string(intended[sentence_index, 0])
            else:
                raw_text = unwrap_mat_string(prompts[sentence_index, 0])
            transcript = normalize_transcript(raw_text.replace("#", ""))
            if not transcript:
                continue

            letter_starts = _trim_step2_sequence(label_payload["letterStarts"][sentence_index], len(transcript))
            letter_durations = _trim_step2_sequence(label_payload["letterDurations"][sentence_index], len(transcript))
            blank_windows = _parse_blank_windows(label_payload["blankWindows"][0, sentence_index])
            time_bins = int(label_lengths[sentence_index]) if int(label_lengths[sentence_index]) > 0 else int(lengths[sentence_index])

            session_records.append(
                SentenceLabelRecord(
                    session_name=session_name,
                    session_id=session_id,
                    sentence_index=sentence_index,
                    block_num=int(blocks[sentence_index]),
                    time_bins=time_bins,
                    transcript=transcript,
                    split=split,
                    letter_starts=letter_starts,
                    letter_durations=letter_durations,
                    blank_windows=blank_windows,
                )
            )

        if max_records_per_session is not None and len(session_records) > max_records_per_session:
            grouped_train = [record for record in session_records if record.split == "train"]
            grouped_val = [record for record in session_records if record.split == "val"]
            n_val = min(len(grouped_val), max(1, max_records_per_session // 3))
            n_train = min(len(grouped_train), max_records_per_session - n_val)
            session_records = grouped_train[:n_train] + grouped_val[:n_val]

        all_records.extend(session_records)

    payload = {
        "session_names": session_names,
        "records": all_records,
        "label_mode": "official",
        "datasets_root": data_root,
        "training_steps_root": training_steps_root,
        "cv_part": cv_part,
    }
    torch.save(
        {**payload, "records": [record.to_serializable() for record in all_records]},
        cache_path,
    )
    return payload


def prepare_paper_v5_label_cache(
    data_root: str,
    cache_path: str,
    val_fraction: float = 0.2,
    hmm_iters: int = 2,
    force_rebuild: bool = False,
    max_records_per_session: Optional[int] = None,
    training_steps_root: Optional[str] = None,
    cv_part: str = "HeldOutBlocks",
) -> Dict[str, object]:
    datasets_root, resolved_training_steps_root = resolve_handwriting_data_roots(
        data_root=data_root,
        training_steps_root=training_steps_root,
    )
    target_official = bool(
        resolved_training_steps_root
        and os.path.isdir(os.path.join(resolved_training_steps_root, "Step2_HMMLabels", cv_part))
    )

    if os.path.exists(cache_path) and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        payload["records"] = [SentenceLabelRecord.from_serializable(item) for item in payload["records"]]
        payload_is_official = payload.get("label_mode") == "official"
        payload_cv_part = payload.get("cv_part", cv_part)
        if payload_is_official == target_official and payload_cv_part == cv_part:
            return payload

    if target_official:
        return prepare_paper_v5_label_cache_official(
            data_root=datasets_root,
            training_steps_root=resolved_training_steps_root,
            cache_path=cache_path,
            cv_part=cv_part,
            force_rebuild=force_rebuild,
            max_records_per_session=max_records_per_session,
        )

    meta_dataset = StanfordSentenceDataset(data_root=datasets_root, bin_size=1)
    train_idx, val_idx = make_block_split_indices(meta_dataset, val_fraction=val_fraction)
    split_map = {idx: "train" for idx in train_idx}
    split_map.update({idx: "val" for idx in val_idx})

    manager = RawSessionDataManager(data_root=datasets_root, cache_size=1)
    session_to_records: Dict[str, List[Tuple[int, object]]] = defaultdict(list)
    for global_index, record in enumerate(meta_dataset.records):
        session_to_records[record.session_name].append((global_index, record))

    all_records: List[SentenceLabelRecord] = []
    for session_name in meta_dataset.session_names:
        grouped = session_to_records[session_name]
        if max_records_per_session is not None:
            grouped_train = [item for item in grouped if split_map.get(item[0], "train") == "train"]
            grouped_val = [item for item in grouped if split_map.get(item[0], "train") == "val"]
            n_val = min(len(grouped_val), max(1, max_records_per_session // 3))
            n_train = min(len(grouped_train), max_records_per_session - n_val)
            grouped = grouped_train[:n_train] + grouped_val[:n_val]

        templates = initialize_templates_from_single_letters(manager, session_name, CHAR_DEF)
        session_records: List[SentenceLabelRecord] = []
        for global_index, record in grouped:
            session_records.append(
                SentenceLabelRecord(
                    session_name=record.session_name,
                    session_id=record.session_id,
                    sentence_index=record.sentence_index,
                    block_num=record.block_num,
                    time_bins=record.time_bins,
                    transcript=record.transcript,
                    split=split_map.get(global_index, "train"),
                    letter_starts=np.zeros((len(record.transcript),), dtype=np.float32),
                    letter_durations=np.zeros((len(record.transcript),), dtype=np.float32),
                    blank_windows=[],
                )
            )

        for _iter_idx in range(hmm_iters):
            labeled_records: List[SentenceLabelRecord] = []
            for record in session_records:
                signal = manager.get_sentence_signal(
                    session_name=record.session_name,
                    sentence_index=record.sentence_index,
                    time_bins=record.time_bins,
                    block_num=record.block_num,
                )
                starts, durations, blank_windows = forced_alignment_labeling(
                    obs=signal,
                    sentence=record.transcript,
                    templates=templates,
                )
                labeled_records.append(
                    SentenceLabelRecord(
                        session_name=record.session_name,
                        session_id=record.session_id,
                        sentence_index=record.sentence_index,
                        block_num=record.block_num,
                        time_bins=record.time_bins,
                        transcript=record.transcript,
                        split=record.split,
                        letter_starts=starts,
                        letter_durations=durations,
                        blank_windows=blank_windows,
                    )
                )

            session_records = labeled_records
            train_records = [record for record in session_records if record.split == "train"]
            templates = update_templates_from_labels(manager=manager, train_records=train_records, templates=templates)

        all_records.extend(session_records)

    payload = {
        "session_names": meta_dataset.session_names,
        "val_fraction": val_fraction,
        "hmm_iters": hmm_iters,
        "records": all_records,
        "label_mode": "generated",
        "datasets_root": datasets_root,
        "training_steps_root": None,
        "cv_part": cv_part,
    }
    torch.save(
        {**payload, "records": [record.to_serializable() for record in all_records]},
        cache_path,
    )
    return payload


def construct_rnn_targets(
    letter_starts: np.ndarray,
    letter_durations: np.ndarray,
    max_time_steps: int,
    sentence: str,
    char_def: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    del letter_durations
    char_prob_target = np.zeros((max_time_steps, len(char_def["charList"])), dtype=np.float32)
    char_start_target = np.zeros((max_time_steps,), dtype=np.float32)
    ignore_error = np.zeros((max_time_steps,), dtype=np.float32)

    if len(sentence) == 0:
        return char_prob_target, char_start_target, ignore_error

    letter_starts = np.asarray(letter_starts, dtype=np.int32)
    for char_pos, char in enumerate(sentence):
        if char_pos < (len(sentence) - 1):
            step_idx = np.arange(letter_starts[char_pos], letter_starts[char_pos + 1], dtype=np.int32)
        else:
            step_idx = np.arange(letter_starts[char_pos], max_time_steps, dtype=np.int32)
        step_idx = step_idx[(step_idx >= 0) & (step_idx < max_time_steps)]
        char_idx = int(char_def["strToCharIdx"][char])
        char_prob_target[step_idx, char_idx] = 1.0

        start_idx = np.arange(letter_starts[char_pos], letter_starts[char_pos] + 21, dtype=np.int32)
        start_idx = start_idx[(start_idx >= 0) & (start_idx < max_time_steps)]
        char_start_target[start_idx] = 1.0

    first_char_idx = int(char_def["strToCharIdx"][sentence[0]])
    prefix = np.arange(0, int(letter_starts[0]), dtype=np.int32)
    prefix = prefix[(prefix >= 0) & (prefix < max_time_steps)]
    char_prob_target[prefix, first_char_idx] = 1.0
    ignore_error[prefix] = 1.0
    return char_prob_target, char_start_target, ignore_error


def collect_word_list(records: Sequence[SentenceLabelRecord]) -> Tuple[List[str], List[int]]:
    word_counter: Dict[str, int] = defaultdict(int)
    for record in records:
        rendered = render_transcript(record.transcript).lower()
        clean = rendered.replace(",", " ").replace(".", " ").replace("?", " ").replace("'", "")
        for word in clean.split():
            word = "".join(ch for ch in word if ch.isalpha())
            if word:
                word_counter[word] += 1

    ordered = sorted(word_counter.items(), key=lambda item: (-item[1], item[0]))
    words = [item[0] for item in ordered]
    rare_word_indices = [idx for idx, word in enumerate(words) if any(ch in word for ch in "xzqj")]
    return words, rare_word_indices


def build_snippet_library(
    data_root: str,
    label_payload: Dict[str, object],
    include_single_letter_snippets: bool = True,
    training_steps_root: Optional[str] = None,
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    session_cache_size = max(1, len(label_payload["session_names"]))
    manager = RawSessionDataManager(data_root=data_root, cache_size=session_cache_size)
    warped_manager = (
        WarpedCubeManager(training_steps_root=training_steps_root, cache_size=session_cache_size)
        if training_steps_root
        else None
    )
    snippets: Dict[int, Dict[str, List[np.ndarray]]] = {}
    for session_id, _session_name in enumerate(label_payload["session_names"]):
        snippets[session_id] = {char_name: [] for char_name in CHAR_DEF["charList"]}
        snippets[session_id]["blank"] = []

    for record in label_payload["records"]:
        if record.split != "train":
            continue
        signal = manager.get_sentence_signal(
            session_name=record.session_name,
            sentence_index=record.sentence_index,
            time_bins=record.time_bins,
            block_num=record.block_num,
        )
        for char_pos, char_abbr in enumerate(record.transcript):
            char_name = CHAR_DEF["abbrToFull"][char_abbr]
            start = int(round(record.letter_starts[char_pos]))
            end = int(record.time_bins) if char_pos == (len(record.transcript) - 1) else int(round(record.letter_starts[char_pos + 1]))
            end = min(end, signal.shape[0])
            if end - start >= 5:
                snippets[record.session_id][char_name].append(signal[start:end, :].astype(np.float32))

        for blank_start, blank_end in record.blank_windows:
            blank_start = max(0, int(blank_start))
            blank_end = min(int(blank_end), signal.shape[0])
            if blank_end - blank_start >= 5:
                snippets[record.session_id]["blank"].append(signal[blank_start:blank_end, :].astype(np.float32))

    if include_single_letter_snippets:
        for session_id, session_name in enumerate(label_payload["session_names"]):
            for char_name, char_len in zip(CHAR_DEF["charList"], CHAR_DEF["charLen"]):
                if warped_manager is not None:
                    for trial in warped_manager.get_char_trials(session_name, char_name):
                        if trial.shape[0] >= 5:
                            snippets[session_id][char_name].append(trial.astype(np.float32))
                else:
                    for trial in manager.get_single_letter_trials(session_name, char_name):
                        start = min(59, max(0, trial.shape[0] - 5))
                        end = min(trial.shape[0], start + int(char_len) + 25)
                        if end - start >= 5:
                            snippets[session_id][char_name].append(trial[start:end, :].astype(np.float32))

    for session_id in snippets.keys():
        if len(snippets[session_id]["blank"]) == 0:
            snippets[session_id]["blank"].append(np.zeros((20, 192), dtype=np.float32))
    return snippets


def generate_synthetic_data_from_snippets(
    session_snippets: Dict[str, List[np.ndarray]],
    word_list: Sequence[str],
    rare_word_indices: Sequence[int],
    n_steps: int,
    blank_prob: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_steps_to_generate = n_steps + 2000
    synth_neural = np.zeros((n_steps_to_generate, 192), dtype=np.float32)
    synth_char_prob = np.zeros((n_steps_to_generate, N_CHAR_CLASSES), dtype=np.float32)
    synth_char_start = np.zeros((n_steps_to_generate,), dtype=np.float32)

    curr_idx = 0
    current_word: List[str] = []
    current_letter_idx = 0
    while curr_idx < n_steps_to_generate:
        if current_letter_idx >= len(current_word):
            current_letter_idx = 0
            current_word = pick_word_for_sentence(word_list, rare_letter_increase=True, rare_word_indices=rare_word_indices)

        current_char = current_word[current_letter_idx]
        if current_char not in CHAR_DEF["strToCharIdx"]:
            current_letter_idx += 1
            continue

        class_idx = int(CHAR_DEF["strToCharIdx"][current_char])
        char_name = CHAR_DEF["charList"][class_idx]
        candidate_list = session_snippets.get(char_name, [])
        if len(candidate_list) == 0:
            current_letter_idx += 1
            continue

        current_snippet = candidate_list[np.random.randint(len(candidate_list))].copy()
        current_snippet = np.nan_to_num(current_snippet, nan=0.0, posinf=0.0, neginf=0.0)
        char_len = current_snippet.shape[0]
        n_steps_for_char = int(round(char_len * 0.7 + np.random.randint(max(1, int(char_len * 0.6)))))
        current_snippet = resample_time_major(current_snippet, max(2, n_steps_for_char))
        current_snippet *= float(0.7 + 0.6 * np.random.rand())

        if np.random.rand() < blank_prob and len(session_snippets.get("blank", [])) > 0:
            blank_snippet = session_snippets["blank"][np.random.randint(len(session_snippets["blank"]))]
            current_snippet = np.concatenate([current_snippet, blank_snippet], axis=0)

        labels = np.zeros((current_snippet.shape[0], N_CHAR_CLASSES), dtype=np.float32)
        labels[:, class_idx] = 1.0

        n_new_steps = current_snippet.shape[0]
        if n_new_steps + curr_idx >= n_steps_to_generate:
            step_limit = n_steps_to_generate - curr_idx
            current_snippet = current_snippet[:step_limit, :]
            labels = labels[:step_limit, :]
            n_new_steps = current_snippet.shape[0]

        synth_neural[curr_idx : curr_idx + n_new_steps, :] = current_snippet
        synth_char_prob[curr_idx : curr_idx + n_new_steps, :] = labels
        synth_char_start[curr_idx : curr_idx + min(20, n_new_steps)] = 1.0

        curr_idx += n_new_steps
        current_letter_idx += 1

    rand_start = np.random.randint(max(1, n_steps_to_generate - n_steps))
    return (
        synth_neural[rand_start : rand_start + n_steps, :],
        synth_char_prob[rand_start : rand_start + n_steps, :],
        synth_char_start[rand_start : rand_start + n_steps],
    )


class PaperV5RealDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        records: Sequence[SentenceLabelRecord],
        time_steps: int = 1200,
        bin_size: int = 2,
        training_steps_root: Optional[str] = None,
        cv_part: str = "HeldOutBlocks",
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.records = list(records)
        self.time_steps = int(time_steps)
        self.bin_size = int(bin_size)
        self.raw_steps = self.time_steps * self.bin_size
        self.edge_space = self.raw_steps - 100
        session_cache_size = max(1, len({record.session_name for record in self.records}))
        self.manager = RawSessionDataManager(data_root=self.data_root, cache_size=session_cache_size)
        self.label_manager = (
            OfficialStepLabelManager(
                training_steps_root=training_steps_root,
                cv_part=cv_part,
                cache_size=session_cache_size,
            )
            if training_steps_root
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        signal = self.manager.get_sentence_signal(
            session_name=record.session_name,
            sentence_index=record.sentence_index,
            time_bins=record.time_bins,
            block_num=record.block_num,
        )
        padded_steps = record.time_bins + self.edge_space * 2
        padded_signal = np.zeros((padded_steps, signal.shape[1]), dtype=np.float32)
        padded_signal[self.edge_space : self.edge_space + record.time_bins, :] = signal

        if self.label_manager is not None:
            raw_char_prob, raw_char_start, raw_ignore_error = self.label_manager.get_sentence_targets(
                session_name=record.session_name,
                sentence_index=record.sentence_index,
                time_bins=record.time_bins,
            )
            char_prob = np.zeros((padded_steps, N_CHAR_CLASSES), dtype=np.float32)
            char_start = np.zeros((padded_steps,), dtype=np.float32)
            err_weights = np.zeros((padded_steps,), dtype=np.float32)
            center_slice = slice(self.edge_space, self.edge_space + record.time_bins)
            char_prob[center_slice, :] = raw_char_prob
            char_start[center_slice] = raw_char_start
            err_weights[center_slice] = 1.0 - raw_ignore_error
        else:
            shifted_starts = record.letter_starts + self.edge_space
            char_prob, char_start, ignore_error = construct_rnn_targets(
                letter_starts=shifted_starts,
                letter_durations=record.letter_durations,
                max_time_steps=padded_steps,
                sentence=record.transcript,
                char_def=CHAR_DEF,
            )
            err_weights = 1.0 - ignore_error
            sentence_end = self.edge_space + record.time_bins
            err_weights[sentence_end:] = 0.0
            if len(shifted_starts) > 0:
                cutoff = min(padded_steps, int(shifted_starts[-1] + 150))
                err_weights[cutoff:] = 0.0

        crop_start = np.random.randint(max(1, padded_steps - self.raw_steps))
        crop_end = crop_start + self.raw_steps
        return {
            "inputs": torch.from_numpy(bin_tensor_time_major(padded_signal[crop_start:crop_end, :], self.bin_size)),
            "char_prob_target": torch.from_numpy(bin_tensor_time_major(char_prob[crop_start:crop_end, :], self.bin_size)),
            "char_start_target": torch.from_numpy(bin_vector(char_start[crop_start:crop_end], self.bin_size)),
            "err_weights": torch.from_numpy(bin_vector(err_weights[crop_start:crop_end], self.bin_size)),
            "input_length": self.time_steps,
            "session_id": record.session_id,
            "session_name": record.session_name,
            "sentence_index": record.sentence_index,
            "block_num": record.block_num,
            "transcript": record.transcript,
            "display_text": render_transcript(record.transcript),
        }


class PaperV5SyntheticDataset(Dataset):
    def __init__(
        self,
        snippet_library: Dict[int, Dict[str, List[np.ndarray]]],
        word_list: Sequence[str],
        rare_word_indices: Sequence[int],
        time_steps: int = 1200,
        bin_size: int = 2,
        n_samples: int = 1000,
    ) -> None:
        super().__init__()
        self.snippet_library = snippet_library
        self.word_list = list(word_list)
        self.rare_word_indices = list(rare_word_indices)
        self.time_steps = int(time_steps)
        self.bin_size = int(bin_size)
        self.raw_steps = self.time_steps * self.bin_size
        self.n_samples = int(n_samples)
        self.session_ids = sorted(snippet_library.keys())

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Dict[str, object]:
        del index
        session_id = random.choice(self.session_ids)
        synth_neural, synth_char_prob, synth_char_start = generate_synthetic_data_from_snippets(
            session_snippets=self.snippet_library[session_id],
            word_list=self.word_list,
            rare_word_indices=self.rare_word_indices,
            n_steps=self.raw_steps,
        )
        err_weights = np.ones((synth_char_start.shape[0],), dtype=np.float32)
        starts = np.argwhere(synth_char_start[1:] - synth_char_start[:-1] >= 0.1).astype(np.int32)
        if len(starts) == 0:
            err_weights[:] = 0.0
        else:
            err_weights[: int(starts[0, 0])] = 0.0

        return {
            "inputs": torch.from_numpy(bin_tensor_time_major(synth_neural, self.bin_size)),
            "char_prob_target": torch.from_numpy(bin_tensor_time_major(synth_char_prob, self.bin_size)),
            "char_start_target": torch.from_numpy(bin_vector(synth_char_start, self.bin_size)),
            "err_weights": torch.from_numpy(bin_vector(err_weights, self.bin_size)),
            "input_length": self.time_steps,
            "session_id": session_id,
            "transcript": "",
            "display_text": "",
        }


class PaperV5EvalDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        records: Sequence[SentenceLabelRecord],
        bin_size: int = 2,
        training_steps_root: Optional[str] = None,
        cv_part: str = "HeldOutBlocks",
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.records = list(records)
        self.bin_size = int(bin_size)
        session_cache_size = max(1, len({record.session_name for record in self.records}))
        self.manager = RawSessionDataManager(data_root=self.data_root, cache_size=session_cache_size)
        self.label_manager = (
            OfficialStepLabelManager(
                training_steps_root=training_steps_root,
                cv_part=cv_part,
                cache_size=session_cache_size,
            )
            if training_steps_root
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        signal = self.manager.get_sentence_signal(
            session_name=record.session_name,
            sentence_index=record.sentence_index,
            time_bins=record.time_bins,
            block_num=record.block_num,
        )
        if self.label_manager is not None:
            char_prob, char_start, ignore_error = self.label_manager.get_sentence_targets(
                session_name=record.session_name,
                sentence_index=record.sentence_index,
                time_bins=record.time_bins,
            )
            err_weights = 1.0 - ignore_error
        else:
            char_prob, char_start, ignore_error = construct_rnn_targets(
                letter_starts=record.letter_starts,
                letter_durations=record.letter_durations,
                max_time_steps=record.time_bins,
                sentence=record.transcript,
                char_def=CHAR_DEF,
            )
            err_weights = 1.0 - ignore_error
            if len(record.letter_starts) > 0:
                cutoff = min(record.time_bins, int(record.letter_starts[-1] + 150))
                err_weights[cutoff:] = 0.0

        binned_signal = bin_tensor_time_major(signal, self.bin_size)
        return {
            "inputs": torch.from_numpy(binned_signal),
            "char_prob_target": torch.from_numpy(bin_tensor_time_major(char_prob, self.bin_size)),
            "char_start_target": torch.from_numpy(bin_vector(char_start, self.bin_size)),
            "err_weights": torch.from_numpy(bin_vector(err_weights, self.bin_size)),
            "input_length": binned_signal.shape[0],
            "session_id": record.session_id,
            "session_name": record.session_name,
            "sentence_index": record.sentence_index,
            "block_num": record.block_num,
            "transcript": record.transcript,
            "display_text": render_transcript(record.transcript),
        }


def collate_fixed_paper_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
        "char_prob_target": torch.stack([item["char_prob_target"] for item in batch], dim=0),
        "char_start_target": torch.stack([item["char_start_target"] for item in batch], dim=0),
        "err_weights": torch.stack([item["err_weights"] for item in batch], dim=0),
        "input_lengths": torch.tensor([int(item["input_length"]) for item in batch], dtype=torch.long),
        "session_ids": torch.tensor([int(item["session_id"]) for item in batch], dtype=torch.long),
        "session_names": [str(item["session_name"]) for item in batch],
        "sentence_indices": torch.tensor([int(item["sentence_index"]) for item in batch], dtype=torch.long),
        "block_nums": torch.tensor([int(item["block_num"]) for item in batch], dtype=torch.long),
        "transcripts": [str(item["transcript"]) for item in batch],
        "display_texts": [str(item["display_text"]) for item in batch],
    }


def collate_variable_paper_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    input_lengths = torch.tensor([int(item["input_length"]) for item in batch], dtype=torch.long)
    max_len = int(input_lengths.max().item())
    batch_size = len(batch)
    inputs = torch.zeros((batch_size, max_len, 192), dtype=torch.float32)
    char_prob = torch.zeros((batch_size, max_len, N_CHAR_CLASSES), dtype=torch.float32)
    char_start = torch.zeros((batch_size, max_len), dtype=torch.float32)
    err_weights = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for row, item in enumerate(batch):
        length = int(item["input_length"])
        inputs[row, :length, :] = item["inputs"]
        char_prob[row, :length, :] = item["char_prob_target"]
        char_start[row, :length] = item["char_start_target"]
        err_weights[row, :length] = item["err_weights"]

    return {
        "inputs": inputs,
        "char_prob_target": char_prob,
        "char_start_target": char_start,
        "err_weights": err_weights,
        "input_lengths": input_lengths,
        "session_ids": torch.tensor([int(item["session_id"]) for item in batch], dtype=torch.long),
        "session_names": [str(item["session_name"]) for item in batch],
        "sentence_indices": torch.tensor([int(item["sentence_index"]) for item in batch], dtype=torch.long),
        "block_nums": torch.tensor([int(item["block_num"]) for item in batch], dtype=torch.long),
        "transcripts": [str(item["transcript"]) for item in batch],
        "display_texts": [str(item["display_text"]) for item in batch],
    }


class GaussianSmoothing1D(nn.Module):
    def __init__(self, channels: int, sigma: float) -> None:
        super().__init__()
        radius = max(1, int(round(sigma * 3)))
        positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-(positions ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        self.padding = radius
        self.register_buffer("kernel", kernel.view(1, 1, -1).repeat(channels, 1, 1))
        self.channels = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        smoothed = F.conv1d(inputs.transpose(1, 2), self.kernel, padding=self.padding, groups=self.channels)
        return smoothed.transpose(1, 2)


class PaperBrainToTextV5(nn.Module):
    def __init__(
        self,
        input_size: int = 192,
        hidden_size: int = 512,
        num_sessions: int = 10,
        skip_len: int = 5,
        dropout: float = 0.1,
        smoothing_sigma: float = 2.0,
        gru1_layers: int = 1,
        gru2_layers: int = 1,
    ) -> None:
        super().__init__()
        eye = torch.eye(input_size, dtype=torch.float32).unsqueeze(0).repeat(num_sessions, 1, 1)
        self.session_weight = nn.Parameter(eye)
        self.session_bias = nn.Parameter(torch.zeros(num_sessions, input_size))
        self.smoother = GaussianSmoothing1D(input_size, sigma=smoothing_sigma)
        self.dropout = nn.Dropout(dropout)
        self.gru1_layers = max(1, int(gru1_layers))
        self.gru2_layers = max(1, int(gru2_layers))
        recurrent_dropout_1 = dropout if self.gru1_layers > 1 else 0.0
        recurrent_dropout_2 = dropout if self.gru2_layers > 1 else 0.0
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.gru1_layers,
            dropout=recurrent_dropout_1,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=self.gru2_layers,
            dropout=recurrent_dropout_2,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, N_CHAR_CLASSES + 1)
        self.start_state_1 = nn.Parameter(torch.zeros(self.gru1_layers, 1, hidden_size))
        self.start_state_2 = nn.Parameter(torch.zeros(self.gru2_layers, 1, hidden_size))
        self.skip_len = int(skip_len)

    def _apply_session_affine(self, inputs: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
        weights = self.session_weight[session_ids]
        bias = self.session_bias[session_ids].unsqueeze(1)
        return torch.bmm(inputs, weights.transpose(1, 2)) + bias

    def _run_gru(self, gru: nn.GRU, x: torch.Tensor, lengths: torch.Tensor, start_state: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        init_state = start_state.repeat(1, x.shape[0], 1)
        packed_out, _ = gru(packed, init_state)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor, session_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._apply_session_affine(inputs, session_ids)
        x = self.smoother(x)
        x = self.dropout(x)
        out1 = self._run_gru(self.gru1, x, input_lengths, self.start_state_1)
        out1 = self.dropout(out1)

        downsampled = out1[:, :: self.skip_len, :]
        down_lengths = torch.div(input_lengths + (self.skip_len - 1), self.skip_len, rounding_mode="floor").clamp_min(1)
        out2 = self._run_gru(self.gru2, downsampled, down_lengths, self.start_state_2)
        out2 = self.dropout(out2)

        logits_down = self.readout(out2)
        logits = torch.repeat_interleave(logits_down, repeats=self.skip_len, dim=1)
        logits = logits[:, : inputs.shape[1], :]
        return logits, input_lengths


def compute_paper_v5_loss(
    logits: torch.Tensor,
    char_prob_target: torch.Tensor,
    char_start_target: torch.Tensor,
    err_weights: torch.Tensor,
    output_delay: int = 50,
    start_loss_weight: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if output_delay > 0 and logits.shape[1] > output_delay:
        logits = logits[:, output_delay:, :]
        char_prob_target = char_prob_target[:, :-output_delay, :]
        char_start_target = char_start_target[:, :-output_delay]
        err_weights = err_weights[:, :-output_delay]

    char_logits = logits[:, :, :-1]
    start_logits = logits[:, :, -1]
    valid = err_weights.sum().clamp_min(1.0)
    log_probs = F.log_softmax(char_logits, dim=-1)
    ce_loss = -(char_prob_target * log_probs).sum(dim=-1)
    ce_loss = (ce_loss * err_weights).sum() / valid

    start_prob = torch.sigmoid(start_logits)
    start_loss = ((start_prob - char_start_target) ** 2 * err_weights).sum() / valid
    total = ce_loss + start_loss_weight * start_loss
    return total, {
        "char_ce": float(ce_loss.item()),
        "start_mse": float(start_loss.item()),
        "valid_steps": float(valid.item()),
    }


def decode_paper_v5_outputs(
    logits: torch.Tensor,
    length: int,
    start_threshold: float = 0.3,
    start_delay: int = 15,
) -> str:
    usable_logits = logits[:length]
    if usable_logits.shape[0] == 0:
        return ""

    char_logits = usable_logits[:, :-1]
    start_prob = torch.sigmoid(usable_logits[:, -1])
    best_class = torch.argmax(char_logits, dim=-1)

    prev_prob = start_prob[:-1]
    next_prob = start_prob[1:]
    transition_idx = torch.argwhere((prev_prob < start_threshold) & (next_prob > start_threshold)).reshape(-1)

    decoded: List[str] = []
    for idx in transition_idx.tolist():
        char_t = min(int(idx) + int(start_delay), usable_logits.shape[0] - 1)
        decoded.append(CHAR_DEF["charListAbbr"][int(best_class[char_t].item())])
    return "".join(decoded)


def evaluate_paper_v5(
    model: PaperBrainToTextV5,
    data_loader: Iterable[Dict[str, object]],
    device: torch.device,
    output_delay: int = 50,
    amp_enabled: bool = False,
    start_threshold: float = 0.3,
    start_delay: int = 15,
    log_every_batches: int = 0,
) -> Dict[str, object]:
    references: List[str] = []
    hypotheses: List[str] = []
    examples: List[Tuple[str, str]] = []
    total_loss = 0.0
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["inputs"].to(device)
            char_prob = batch["char_prob_target"].to(device)
            char_start = batch["char_start_target"].to(device)
            err_weights = batch["err_weights"].to(device)
            lengths = batch["input_lengths"].to(device)
            session_ids = batch["session_ids"].to(device)

            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                logits, _ = model(inputs, lengths, session_ids)
                loss, _ = compute_paper_v5_loss(
                    logits=logits,
                    char_prob_target=char_prob,
                    char_start_target=char_start,
                    err_weights=err_weights,
                    output_delay=output_delay,
                )
            total_loss += float(loss.item())
            n_batches += 1
            if log_every_batches > 0 and (n_batches % log_every_batches) == 0:
                total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
                if total_batches is not None:
                    print(f"           validation progress: {n_batches}/{total_batches} batches", flush=True)
                else:
                    print(f"           validation progress: {n_batches} batches", flush=True)

            for row in range(inputs.shape[0]):
                hyp = decode_paper_v5_outputs(
                    logits[row].cpu(),
                    int(lengths[row].item()),
                    start_threshold=start_threshold,
                    start_delay=start_delay,
                )
                ref = batch["transcripts"][row]
                references.append(ref)
                hypotheses.append(hyp)
                if len(examples) < 5:
                    examples.append((render_transcript(ref), render_transcript(hyp)))

    metrics = compute_error_rates(references, hypotheses)
    metrics["loss"] = total_loss / max(1, n_batches)
    metrics["examples"] = examples
    return metrics
