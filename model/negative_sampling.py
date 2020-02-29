from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

from .utils import Sampler


class NegativeSampling(nn.Module):
    def __init__(
            self,
            embeddings: nn.Module,
            word_freqs: List[int],
            alpha: float = 0.75,
            num_negative: int = 10
    ):
        super(NegativeSampling, self).__init__()
        self.embeddings = embeddings
        self.word_freqs = word_freqs
        self.num_negative = num_negative
        self.sampler = Sampler(word_freqs, alpha)
        self.activate_func = nn.LogSigmoid()

    def get_negative_embeddings(self, shape: Tuple[int]):
        sample_ids = self.sampler.sample(shape=shape)
        sample_ids = torch.tensor(sample_ids, dtype=torch.long, device=self.embeddings.device())
        negative_embeddings = self.embeddings(sample_ids)
        return negative_embeddings

    def forward(self, target_embs: torch.Tensor, context_tensor: torch.Tensor):
        """

        Args:
            target_embs: Tensor with shape (batch_size, seq_len, emb_size)
            context_tensor:  Tensor with shape (batch_size, emb_size)

        Returns:

        """
        batch_size, seq_len, _ = target_embs.size()
        positive_samples_loss = self.activate_func((target_embs * context_tensor)).sum(dim=-1)

        negative_samples = self.get_negative_embeddings(shape=(batch_size, seq_len, self.num_negative))
        negative_samples_loss = self.activate_func((-negative_samples * context_tensor.unsqueeze(-1)).sum(-1)).sum(-1)
        return -(positive_samples_loss + negative_samples_loss).sum()
