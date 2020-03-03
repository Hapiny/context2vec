from collections import defaultdict
from typing import List
import unicodedata
from tqdm import tqdm


class WikiText103Processing:
    def __init__(
            self,
            tokens_path: str,
            max_sentence_length: int = 64,
            lower: bool = False,
    ):
        self.path = tokens_path
        self.save_path = tokens_path + ".processed"
        self.max_sentence_length = max_sentence_length
        self.lower = lower
        self.unk_token = "<unk>"

    @staticmethod
    def strip_accents(s: str) -> str:
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    def process_wiki_paragraph(self, tokens: List[str]) -> List[List[str]]:
        sentences, sentence = [], []
        sentence_len = 0
        for token_idx, token in enumerate(tokens):
            if sentence_len >= self.max_sentence_length or token == ".":
                if token == ".":
                    sentence += token
                sentences.append(sentence)
                sentence, sentence_len = [], 0
                if token == ".":
                    continue

            token = self.strip_accents(token)
            if self.lower:
                token = token.lower()
            if token == ".":
                sentence.append(token)
                break
            if not token.isascii():
                token = self.unk_token
            sentence.append(token)
            sentence_len += 1
        return sentences

    def iter_lines(self):
        with open(self.path, "r") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                line = line.strip()
                if not line or line.startswith("="):
                    continue
                yield line

    def process_dataset(self):
        save_file = open(self.save_path, "w")
        counter = defaultdict(int)
        try:
            for line in tqdm(self.iter_lines(), desc=f"Processing file: {self.path}"):
                tokens = line.lower().split(" ")
                sentences = self.process_wiki_paragraph(tokens=tokens)
                for sentence in sentences:
                    for token in sentence:
                        counter[token] += 1
                    save_file.write(" ".join(sentence) + "\n")
        except Exception as e:
            print(e)
            save_file.close()
        with open(self.path + ".counter", "w") as f:
            for word, count in counter.items():
                f.write(f"{word}\t{count}\n")


if __name__ == "__main__":
    processor = WikiText103Processing(tokens_path="./dataset/wikitext-103/wiki.train.tokens")
    processor.process_dataset()

    processor = WikiText103Processing(tokens_path="./dataset/wikitext-103/wiki.valid.tokens")
    processor.process_dataset()

    processor = WikiText103Processing(tokens_path="./dataset/wikitext-103/wiki.test.tokens")
    processor.process_dataset()
