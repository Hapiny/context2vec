import torch
import torch.nn as nn

from typing import List, Union, Tuple, Optional
from overrides import overrides

from modeling.utils import Sampler
from modeling.mlp import MLP
from modeling.negative_sampling import NegativeSampling


class Context2Vec(nn.Module):
    def __init__(
            self,
            word_freqs: List[int],
            context_emb_size: int = 512,
            target_emb_size: int = 512,
            lstm_size: int = 256,
            lstm_num_layers: int = 1,
            mlp_hidden_size: int = 256,
            mlp_num_layers: Union[List[int], int] = 2,
            mlp_dropout: float = 0.0,
            mlp_activate_func: str = "relu",
            num_negative_samples: int = 10,
            alpha: float = 0.75,
            device: torch.device = None
    ):
        super(Context2Vec, self).__init__()
        self.vocab_size = len(word_freqs)
        self.context_emb_size = context_emb_size
        self.target_emb_size = target_emb_size
        self.num_negative_samples = num_negative_samples
        if device is None:
            device = torch.device("cpu")
        self.device = device

        # Create embedding matrix for Target words
        self.target_embs = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.target_emb_size,
            padding_idx=0,
        )

        # Create embedding matrices for Left and Right context words
        context_embs = self.create_context_embeddings(self.vocab_size, self.context_emb_size)
        self.left_context_embs, self.right_context_embs = context_embs

        # Create two separate LSTMs for Left and Right context
        lstm_nets = self.create_lstm(hidden_size=lstm_size, num_layers=lstm_num_layers)
        self.left_context_lstm, self.right_context_lstm = lstm_nets

        # Create Multi-Layer Perceptron for obtaining context vector
        self.mlp = MLP(
            input_size=2 * lstm_size,
            hidden_size=mlp_hidden_size,
            output_size=target_emb_size,
            num_layers=mlp_num_layers,
            dropout_rate=mlp_dropout,
            activate_func=mlp_activate_func
        )

        self.sampler = Sampler(word_freqs=word_freqs, alpha=alpha)

        # Create Negative Sampling Loss function
        self.loss = NegativeSampling()

        # Move modeling to given device
        self.to(device=device)

    @staticmethod
    def create_context_embeddings(vocab_size: int, emb_size: int = 512) -> Tuple[nn.Module, nn.Module]:
        left_context_embs = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            padding_idx=0,
        )
        right_context_embs = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            padding_idx=0,
        )
        return left_context_embs, right_context_embs

    def create_lstm(self, hidden_size: int, num_layers: int = 1) -> Tuple[nn.Module, nn.Module]:
        left_context_lstm = nn.LSTM(
            input_size=self.context_emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        right_context_lstm = nn.LSTM(
            input_size=self.context_emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        return left_context_lstm, right_context_lstm

    def sample(self, shape: torch.Size) -> torch.Tensor:
        """
        Method that samples negative target words and then obtains its word embeddings.
        Args:
            shape: Shape of the matrix with negative target word indexes, e.g.
                if shape is (2, 3, 5) and dimensions corresponds to
                (batch_size, sequence_len, num_negative) this means that we want
                to get 5 negative samples for each of 3 timestamps in each of 2 examples
                from the batch.

        Returns:
            negative_embeddings: Embeddings corresponding to sampled negative target words.
        """
        sample_ids = self.sampler(size=shape)
        negative_embeddings = self.target_embs(sample_ids)
        return negative_embeddings

    @overrides
    def forward(self, input_tensor: torch.Tensor, target_ids: Optional[List[int]] = None):
        left_context_ids = input_tensor[:, :-1]
        right_context_ids = input_tensor.flip(-1)[:, :-1]

        left_context_embs = self.left_context_embs(left_context_ids)
        right_context_embs = self.right_context_embs(right_context_ids)
        target_embs = self.target_embs(input_tensor[:, 1:-1])

        left_hidden_states, _ = self.left_context_lstm(left_context_embs)
        left_hidden_states = left_hidden_states[:, :-1]

        right_hidden_states, _ = self.right_context_lstm(right_context_embs)
        right_hidden_states = right_hidden_states[:, :-1].flip(-1)

        bi_dir_hidden_state = torch.cat((left_hidden_states, right_hidden_states), dim=-1)
        context_tensor = self.mlp(bi_dir_hidden_state)

        negative_shape = torch.Size((target_embs.size(0), target_embs.size(1), self.num_negative_samples))
        negative_samples = self.sample(shape=negative_shape)
        loss = self.loss(target_embs, negative_samples, context_tensor)
        return loss
