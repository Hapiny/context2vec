import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from typing import List, Union, Tuple
from overrides import overrides

from modeling.utils import Sampler
from modeling.mlp import MLP
from modeling.negative_sampling import NegativeSampling


class Context2Vec(nn.Module):
    def __init__(
            self,
            word_freqs: List[int],
            pad_token_id: int,
            context_emb_size: int = 300,
            target_emb_size: int = 600,
            lstm_size: int = 600,
            lstm_num_layers: int = 1,
            mlp_size: Union[List[int], int] = 1200,
            mlp_num_layers: int = 2,
            mlp_dropout: float = 0.0,
            num_negative_samples: int = 10,
            alpha: float = 0.75,
            device: torch.device = None,
            summary_writer: SummaryWriter = None
    ):
        super(Context2Vec, self).__init__()
        self.pad_token_id = pad_token_id
        self.vocab_size = len(word_freqs)
        self.context_emb_size = context_emb_size
        self.target_emb_size = target_emb_size
        self.num_negative_samples = num_negative_samples
        self.writer = summary_writer
        if device is None:
            device = torch.device("cpu")
        self.device = device

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
        lstm_nets = self.create_lstm(hidden_size=lstm_size, num_layers=lstm_num_layers)
        self.left_context_lstm, self.right_context_lstm = lstm_nets

        # Create Multi-Layer Perceptron for obtaining context vector
        self.mlp = MLP(
            input_size=2 * lstm_size,
            hidden_size=mlp_size,
            output_size=target_emb_size,
            num_layers=mlp_num_layers,
            dropout_rate=mlp_dropout,
        )

        self.sampler = Sampler(word_freqs=word_freqs, alpha=alpha)

        # Create Negative Sampling Loss function
        self.loss = NegativeSampling()

        # Initialize context embeddings
        std = (1. / self.context_emb_size) ** 0.5
        self.left_context_embs.weight.data.normal_(0, std)
        self.right_context_embs.weight.data.normal_(0, std)
        self.target_embs.weight.data.zero_()

        # Move modeling to given device
        self.to(device=device)

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
        sample_ids = self.sampler(size=shape).to(self.device)
        negative_embeddings = self.target_embs(sample_ids)
        return negative_embeddings

    def get_context_vector(self, input_tensor: torch.Tensor) -> torch.Tensor:
        left_context_ids = input_tensor[:, :-1]
        right_context_ids = input_tensor.flip(1)[:, :-1]

        left_context_embs = self.left_context_embs(left_context_ids)
        right_context_embs = self.right_context_embs(right_context_ids)

        left_hidden_states, _ = self.left_context_lstm(left_context_embs)
        left_hidden_states = left_hidden_states[:, :-1, :]

        right_hidden_states, _ = self.right_context_lstm(right_context_embs)
        right_hidden_states = right_hidden_states[:, :-1, :].flip(1)

        bi_dir_hidden_state = torch.cat((left_hidden_states, right_hidden_states), dim=2)
        context_tensor = self.mlp(bi_dir_hidden_state)
        return context_tensor

    def get_closest_words_to_context(self, context_ids: torch.Tensor, target_pos: int, k: int = 10):
        context_ids = context_ids.t()
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
        negative_shape = torch.Size((target_embs.size(0), target_embs.size(1), self.num_negative_samples))
        negative_samples = self.sample(shape=negative_shape)
        loss = self.loss(target_embs, negative_samples, context_tensor)
        return loss
