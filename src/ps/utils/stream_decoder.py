"""把 FPGA 实际返回的 logit 分块解码为字符，保持既有事件解码规则。"""

from __future__ import annotations

import math

VOCAB = "abcdefghijklmnopqrstuvwxyz>,'~?"


def display_text(raw: str) -> str:
    """将模型词表中的空格和句号标记转换为展示文字。"""
    return raw.replace(">", " ").replace("~", ".")


class StreamingDecoder:
    """保留跨分块状态，等待起始事件后第 15 帧再提交字符。"""

    def __init__(self, length: int, threshold: float = 0.3, delay: int = 15):
        self.length = length
        self.threshold = threshold
        self.delay = delay
        self.position = 0
        self.previous_onset = None
        self.pending = []
        self.raw = ""

    def feed(self, rows) -> str:
        """每个硬件输出帧重复 5 次；只用已返回的结果生成字符。"""
        for row in rows:
            onset_logit = max(-40.0, min(40.0, float(row[-1]) / 256.0))
            onset = 1.0 / (1.0 + math.exp(-onset_logit))
            char_index = max(range(31), key=lambda index: int(row[index]))
            for _ in range(5):
                if self.position >= self.length:
                    break
                if (self.previous_onset is not None
                        and self.previous_onset < self.threshold < onset):
                    target = min(self.position - 1 + self.delay, self.length - 1)
                    self.pending.append(target)
                while self.pending and self.pending[0] <= self.position:
                    self.raw += VOCAB[char_index]
                    self.pending.pop(0)
                self.previous_onset = onset
                self.position += 1
        return display_text(self.raw)
