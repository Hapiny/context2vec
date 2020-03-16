from typing import List, Tuple

import numpy as np
import torch.nn as nn
import torch
from overrides import overrides


class Sampler(nn.Module):
    def __init__(self, word_freqs: np.ndarray, alpha: float = 0.75):
        super(Sampler, self).__init__()
        self.word_freqs = word_freqs
        self.alpha = alpha
        self.threshold, self.values = self.init_thresholds()

    def init_thresholds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        vocab_size = len(self.word_freqs)
        probs = np.array(self.word_freqs, dtype=np.float32)
        probs = np.power(probs, self.alpha)
        probs /= probs.sum()

        threshold = np.zeros(shape=(vocab_size,), dtype=np.float32)
        values = np.zeros(shape=(vocab_size * 2,), dtype=np.int32)

        il, ir = 0, 0
        pairs = list(enumerate(probs))
        pairs.sort(key=lambda pair: pair[1])
        for idx, prob in pairs:
            p = prob * vocab_size
            while p > 1 and ir < il:
                values[ir * 2 + 1] = idx
                p -= 1.0 - threshold[ir]
                ir += 1
            threshold[il] = p
            values[il * 2] = idx
            il += 1

        threshold = torch.from_numpy(threshold).type(torch.float)
        values = torch.from_numpy(values).type(torch.long)
        return threshold, values

    @overrides
    def forward(self, size: torch.Size) -> torch.Tensor:
        p = torch.rand(size=size) * self.threshold.size()[0]
        index = p.type(torch.long)
        left_right = (self.threshold[index] < p - index).type(torch.long)
        return self.values[index * 2 + left_right]
