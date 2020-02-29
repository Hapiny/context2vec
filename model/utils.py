from typing import List, Tuple

import numpy as np


class Sampler:
    def __init__(self, word_freqs: List[int], alpha: float = 0.75):
        self.word_freqs = word_freqs
        self.alpha = alpha
        self.threshold, self.values = self.init_thresholds()

    def init_thresholds(self) -> Tuple[np.ndarray, np.ndarray]:
        vocab_size = len(self.word_freqs)
        probs = np.array(self.word_freqs, dtype=np.float32)
        probs = np.power(probs, self.alpha)
        probs /= probs.sum()

        threshold = np.zeros(shape=(vocab_size,), dtype=np.float32)
        values = np.zeros(shape=(vocab_size * 2, ), dtype=np.int32)

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

        return threshold, values

    def sample(self, shape: Tuple[int]):
        ps = np.random.uniform(0, 1, shape)
        pb = ps * self.threshold.shape[0]
        index = pb.astype(np.int32)
        left_right = (self.threshold[index] < pb - index).astype(np.int32)
        return self.values[index * 2 + left_right]
