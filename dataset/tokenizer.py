from typing import Dict, Optional, List, Union, Tuple

import torch
from torchtext.vocab import Vocab

DEFAULT_VOCAB_SIZE = int(1e5)
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>"
]


class Tokenizer:
    def __init__(
            self,
            counter_path: str = None,
            word2id: Dict[str, int] = None,
            vocab_size: int = DEFAULT_VOCAB_SIZE,
            min_freq: int = 3
    ):
        assert counter_path is not None or word2id is not None
        self.counter_path = counter_path
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2id = None
        self.counter = None
        if word2id is not None:
            self.word2id = word2id
        else:
            self.word2id = self.create_vocab()
        self.vocab = list(self.word2id.keys())
        self.special_tokens = SPECIAL_TOKENS
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    @staticmethod
    def tokenize(s: str) -> List[str]:
        return s.split(" ")

    def convert_tokens_to_ids(
            self,
            tokens: List[List[str]],
            device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        if self.word2id == dict():
            assert "Word to index mapping not set yet."
        max_len = 0
        converted = []
        # Convert to ids
        for toks in tokens:
            cur_len = len(toks)
            max_len = max(cur_len, max_len)
            converted.append([self.word2id.get(token, self.unk_token_id) for token in toks])

        # Pad for batch
        lengths = []
        for i, ids in enumerate(converted):
            cur_len = len(ids)
            lengths.append(cur_len + 1)
            padded = [self.bos_token_id] + ids + [self.eos_token_id] + [self.pad_token_id for _ in range(max_len - cur_len)]
            converted[i] = padded
        ids_tensor = torch.tensor(converted, dtype=torch.long, device=device)
        return ids_tensor, lengths

    def create_vocab(self, save_path: Optional[str] = None) -> Dict[str, int]:
        counter = dict()
        with open(self.counter_path, "r") as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                word, count = parts
                counter[word] = int(count)
        for token in SPECIAL_TOKENS:
            if token not in counter:
                counter[token] = 0

        vocab = Vocab(
            counter=counter,
            max_size=self.vocab_size,
            min_freq=self.min_freq,
            specials=SPECIAL_TOKENS,
            specials_first=True
        )
        word2id = vocab.stoi
        self.counter = dict()
        for word in word2id:
            self.counter[word] = counter[word]

        if save_path is not None:
            with open(save_path, "w") as fp:
                for word, idx in word2id.items():
                    fp.write(f"{word}")
        return word2id
