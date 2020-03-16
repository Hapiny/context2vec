from typing import List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides
from torch.utils.tensorboard import SummaryWriter

from .mlp import MLP


class Context2Vec(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            pad_token_id: int,
            encoder_type: str = "lstm",
            context_emb_size: int = 300,
            target_emb_size: int = 600,
            lstm_size: int = 600,
            lstm_num_layers: int = 1,
            mlp_size: Union[List[int], int] = 1200,
            mlp_num_layers: int = 2,
            mlp_dropout: float = 0.0,
            summary_writer: SummaryWriter = None
    ):
        super(Context2Vec, self).__init__()
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.context_emb_size = context_emb_size
        self.target_emb_size = target_emb_size
        self.writer = summary_writer
        self.encoder_type = encoder_type

        # Create embedding matrix for Target words
        self.target_embs = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.target_emb_size,
            padding_idx=self.pad_token_id,
        )

        # Create embedding matrices for Left and Right context words
        context_embs = self.create_context_embeddings(self.vocab_size, self.context_emb_size)
        self.left_context_embs, self.right_context_embs = context_embs

        # Create two separate LSTMs for Left and Right context
        if self.encoder_type == "lstm":
            lstm_nets = self.create_lstm(hidden_size=lstm_size, num_layers=lstm_num_layers)
            self.left_context_lstm, self.right_context_lstm = lstm_nets
        elif self.encoder_type == "transformer":
            self.left_context_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.context_emb_size, nhead=8),
                num_layers=2
            )
            self.right_context_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.context_emb_size, nhead=8),
                num_layers=2
            )
        else:
            raise ValueError(f"Invalid encoder type")

        # Create Multi-Layer Perceptron for obtaining context vector
        self.mlp = MLP(
            input_size=(2 * lstm_size if self.encoder_type == "lstm" else 2 * self.context_emb_size),
            hidden_size=mlp_size,
            output_size=target_emb_size,
            num_layers=mlp_num_layers,
            dropout_rate=mlp_dropout,
        )

        # Initialize context embeddings
        std = (1. / self.context_emb_size) ** 0.5
        self.left_context_embs.weight.data.normal_(0, std)
        self.right_context_embs.weight.data.normal_(0, std)
        self.target_embs.weight.data.zero_()

    def create_context_embeddings(self, vocab_size: int, emb_size: int = 512) -> Tuple[nn.Module, nn.Module]:
        left_context_embs = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            padding_idx=self.pad_token_id,
        )
        right_context_embs = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            padding_idx=self.pad_token_id,
        )
        return left_context_embs, right_context_embs

    def create_lstm(self, hidden_size: int, num_layers: int = 1) -> Tuple[nn.Module, nn.Module]:
        left_context_lstm = nn.LSTM(
            input_size=self.context_emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
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
        sample_ids = self.sampler(size=shape).type(torch.long)
        sample_ids = sample_ids.to(self.target_embs.weight.data.device)
        negative_embeddings = self.target_embs(sample_ids)
        return negative_embeddings

    def get_context_vector(self, input_tensor: torch.Tensor) -> torch.Tensor:
        left_context_ids = input_tensor[:, :-1]
        right_context_ids = input_tensor.flip(1)[:, :-1]

        left_context_embs = self.left_context_embs(left_context_ids)
        right_context_embs = self.right_context_embs(right_context_ids)

        if self.encoder_type == "lstm":
            left_hidden_states, _ = self.left_context_lstm(left_context_embs)
            right_hidden_states, _ = self.right_context_lstm(right_context_embs)
        elif self.encoder_type == "transformer":
            left_hidden_states = self.left_context_encoder(left_context_embs)
            right_hidden_states = self.right_context_encoder(right_context_embs)
        else:
            raise ValueError("Invalid encoder type")

        left_hidden_states = left_hidden_states[:, :-1, :]
        right_hidden_states = right_hidden_states[:, :-1, :].flip(1)

        bi_dir_hidden_state = torch.cat((left_hidden_states, right_hidden_states), dim=2)
        context_tensor = self.mlp(bi_dir_hidden_state)
        return context_tensor

    def get_closest_words_to_context(self, context_ids: torch.Tensor, target_pos: int, k: int = 10):
        # context_ids = context_ids.t()
        context_vector = self.get_context_vector(context_ids)[0, target_pos, :]
        logits = (self.target_embs.weight.data * context_vector).sum(-1)
        probs = F.softmax(logits, dim=0)
        top_vals, top_ids = probs.topk(k=k)
        top_ids = top_ids.tolist()
        top_vals = top_vals.tolist()
        return top_ids, top_vals

    @overrides
    def forward(self, input_tensor: torch.Tensor):
        target_embs = self.target_embs(input_tensor[:, 1:-1])
        context_tensor = self.get_context_vector(input_tensor)
        return context_tensor, target_embs
