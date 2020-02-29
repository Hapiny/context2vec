import torch
import torch.nn as nn

from typing import List, Union, Tuple

from .mlp import MLP
from .negative_sampling import NegativeSampling


class Context2Vec(nn.Module):
    def __init__(
            self,
            word_freqs: List[int],
            context_emb_size: int,
            target_emb_size: int,
            lstm_size: int,
            lstm_num_layers: int = 1,
            mlp_hidden_size: int = 256,
            mlp_num_layers: Union[List[int], int] = 2,
            mlp_dropout: float = 0.0,
            mlp_activate_func: str = "relu",
            alpha: float = 0.75,
            device: torch.device = None
    ):
        super(Context2Vec, self).__init__()
        self.vocab_size = len(word_freqs)
        self.context_emb_size = context_emb_size
        self.target_emb_size = target_emb_size

        # Create embedding matrix for Target words
        self.target_embs = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.target_emb_size,
            padding_idx=0
        )

        # Create embedding matrices for Left and Right context words
        context_embs = self.create_context_embeddings(vocab_size, context_emb_size)
        self.left_context_embs, self.right_context_embs = context_embs

        # Create two separate LSTMs for Left and Right context
        lstm_nets = self.create_lsmt(hidden_size=lstm_size, num_layers=lstm_num_layers)
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

        # Create Negative Sampling Loss function
        self.loss = NegativeSampling(word_freqs, alpha=alpha)

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

    def forward(self, input_ids: List[List[int]], target_ids: List[int]):
        input_ids = torch.tensor(input_ids, device=self.device)
        return input_ids
