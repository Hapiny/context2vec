import numpy as np
import torch
import torch.nn as nn
from overrides import overrides

from .utils import Sampler


class NegativeSampling(nn.Module):
    def __init__(
            self,
            embeddings: nn.Module,
            freqs: np.ndarray,
            num_negative_samples: int = 10,
            alpha: float = 0.75
    ):
        """
        Class that implements Negative Sampling Loss function for context2vec modeling.
        """
        super(NegativeSampling, self).__init__()
        self.activate_func = nn.LogSigmoid()
        self.embeddings = embeddings
        self.num_negative_samples = num_negative_samples
        self.sampler = Sampler(word_freqs=freqs, alpha=alpha)

    @overrides
    def forward(
            self,
            positive_sample: torch.Tensor,
            context_tensor: torch.Tensor
    ):
        """
        Method that computes Negative Sampling loss.
        Args:
            positive_sample: Tensor with shape (seq_len, batch_size, emb_size)
            context_tensor:  Tensor with shape (seq_len, batch_size, emb_size)

        Returns:

        """
        negative_shape = torch.Size((positive_sample.size(0), positive_sample.size(1), self.num_negative_samples))
        negative_sample_ids = self.sampler(size=negative_shape).type(torch.long).to(positive_sample.device)
        negative_samples = self.embeddings(negative_sample_ids)
        if isinstance(context_tensor, tuple):
            context_tensor = torch.cat(context_tensor, dim=0)
        positive_logits = (positive_sample * context_tensor).sum(dim=-1)
        positive_samples_loss = self.activate_func(positive_logits)
        logits = (-negative_samples * context_tensor.unsqueeze(2)).sum(-1)
        negative_samples_loss = self.activate_func(logits).sum(-1)
        loss = -(positive_samples_loss + negative_samples_loss).sum()
        return loss
