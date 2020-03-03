import torch
from typing import Optional
from dataset.tokenizer import Tokenizer
from numpy import ceil


class Batcher:
    def __init__(
            self,
            dataset_path: str,
            tokenizer: Tokenizer,
            batch_size: int = 64,
            device: Optional[torch.device] = None
    ):
        self.dataset_path = dataset_path
        self.dataset_size = 0
        self.dataset_idx = 0
        with open(self.dataset_path, "r") as f:
            for _ in f:
                self.dataset_size += 1

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        num_batches = int(ceil(self.dataset_size / self.batch_size))
        return num_batches

    def __iter__(self):
        batch_of_texts = []
        with open(self.dataset_path, "r") as fp:
            for i, line in enumerate(fp):
                if len(batch_of_texts) == self.batch_size or i == self.dataset_size - 1:
                    tokens = [self.tokenizer.tokenize(text) for text in batch_of_texts]
                    input_ids, lengths = self.tokenizer.convert_tokens_to_ids(tokens, device=self.device)
                    batch_of_texts = []
                    sorted_ids, sorted_values = list(zip(*sorted(enumerate(lengths), reverse=True, key=lambda x: x[1])))
                    yield input_ids[list(sorted_ids)].t(), sorted_values
                batch_of_texts.append(line.strip())
                self.dataset_idx += 1
            self.dataset_idx = 0
