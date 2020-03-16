import torch
import torch.nn as nn
from overrides import overrides


class NegativeSampling(nn.Module):
    def __init__(self):
        """
        Class that implements Negative Sampling Loss function for context2vec modeling.
        """
        super(NegativeSampling, self).__init__()
        self.activate_func = nn.LogSigmoid()

    @overrides
    def forward(
            self,
            positive_sample: torch.Tensor,
            negative_samples: torch.Tensor,
            context_tensor: torch.Tensor
    ):
        """
        Method that computes Negative Sampling loss.
        Args:
            negative_samples: Tensor with shape (seq_len, batch_size, num_negative, emb_size)
            positive_sample: Tensor with shape (seq_len, batch_size, emb_size)
            context_tensor:  Tensor with shape (seq_len, batch_size, emb_size)

        Returns:

        """
        positive_logits = (positive_sample * context_tensor).sum(dim=-1)
        positive_samples_loss = self.activate_func(positive_logits)
        logits = (-negative_samples * context_tensor.unsqueeze(2)).sum(-1)
        negative_samples_loss = self.activate_func(logits).sum(-1)
        loss = -(positive_samples_loss + negative_samples_loss).sum()
        return loss
