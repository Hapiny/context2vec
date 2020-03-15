from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from torchtext import data
from torchtext.vocab import Vocab
from tqdm import tqdm

DEFAULT_VOCAB_SIZE = 150_000
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>"
]


class Dataset:
    def __init__(
            self,
            sentences: np.ndarray,
            vocab: Vocab = None,
            batch_size: int = 64,
            min_freq: int = 3,
            vocab_size: int = DEFAULT_VOCAB_SIZE,
            device: torch.device = None,
            is_train: bool = False
    ):
        self.sent_dict = self.gather_by_lengths(sentences)

        self.pad_token = SPECIAL_TOKENS[0]
        self.unk_token = SPECIAL_TOKENS[1]
        self.bos_token = SPECIAL_TOKENS[2]
        self.eos_token = SPECIAL_TOKENS[3]

        self.batch_size = batch_size
        self.is_train = is_train
        self.device = device

        self.sentence_field = data.Field(
            use_vocab=True,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            init_token=self.bos_token,
            eos_token=self.eos_token,
            batch_first=True,
            include_lengths=False
        )
        self.sentence_id_field = data.Field(use_vocab=False, batch_first=True)

        # Create vocabulary or use given vocab
        if vocab is None:
            self.sentence_field.build_vocab(
                sentences,
                max_size=vocab_size,
                min_freq=min_freq
            )
        else:
            self.sentence_field.vocab = vocab

        self.vocab = self.sentence_field.vocab
        self.freqs = np.array(
            [
                self.vocab.freqs[word]
                if word in self.vocab.freqs and word not in SPECIAL_TOKENS
                else 0
                for word in self.vocab.itos
            ]
        )

        self.dataset = self.create_dataset(self.sent_dict, sentences)
        self.pad_token_id = self.vocab.stoi[self.pad_token]
        self.unk_token_id = self.vocab.stoi[self.unk_token]
        self.bos_token_id = self.vocab.stoi[self.bos_token]
        self.eos_token_id = self.vocab.stoi[self.eos_token]

    def convert_ids_to_tokens(self, ids: List[List[int]]):
        return [
            [self.vocab.itos[idx] for idx in sentence_ids]
            for sentence_ids in ids
        ]

    def convert_tokens_to_ids(self, tokens: List[List[str]]):
        ids = [
            [self.bos_token_id] +
            [self.vocab.stoi.get(token, self.unk_token_id) for token in sentence_tokens] +
            [self.eos_token_id]
            for sentence_tokens in tokens
        ]
        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        return ids

    @staticmethod
    def gather_by_lengths(sentences: np.ndarray):
        lengths = [(index, len(sent)) for index, sent in enumerate(sentences)]
        lengths = sorted(lengths, key=lambda x: x[1], reverse=True)

        sent_dict = defaultdict(list)
        for index, length in lengths:
            if length > 64:
                continue
            sent_dict[length].append(index)

        return sent_dict

    def create_dataset(self, sent_dict: Dict[int, List[int]], sentences: np.ndarray):
        datasets = dict()
        fields = [
            ("sentence", self.sentence_field),
            ("id", self.sentence_id_field)
        ]
        for sent_length, sent_indices in sent_dict.items():
            sent_indices = np.array(sent_indices)
            items = [*zip(sentences[sent_indices], sent_indices[:, np.newaxis])]
            datasets[sent_length] = data.Dataset(self.get_examples(items, fields), fields)
        np.random.seed(777)
        return np.random.permutation(list(datasets.values()))

    @staticmethod
    def get_examples(items: List, fields: List):
        return [data.Example.fromlist(item, fields) for item in items]

    def get_batch_iter(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        for dataset in tqdm(self.dataset):
            dataset_iterator = data.Iterator(
                dataset=dataset,
                batch_size=batch_size,
                train=self.is_train,
                repeat=False,
                device=self.device,
            )
            for batch in dataset_iterator:
                yield batch
