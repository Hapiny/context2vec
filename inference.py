import os
from modeling.context2vec import Context2Vec
from dataset.tokenizer import Tokenizer
import torch

epoch = 30
model_path = f"./experiments/wikitext-103/c2v_{epoch}"
counter_path = "./dataset/wikitext-103/wiki.train.tokens.counter"
dataset_path = "./dataset/wikitext-103/wiki.train.tokens.processed"

tokenizer = Tokenizer(counter_path=counter_path)

device = torch.device("cuda:3")
model = Context2Vec(
    context_emb_size=300,
    target_emb_size=600,
    word_freqs=list(tokenizer.counter.values()),
    lstm_size=600,
    mlp_num_layers=2,
    mlp_hidden_size=1200,
    mlp_dropout=0.1,
    alpha=0.75,
    num_negative_samples=10,
    device=device
)
model.load_state_dict(torch.load(model_path))
model.eval()

while True:
    tokens = [input("Input sentence: >>> ").strip().split()]
    target_pos = int(input("Input target position: >>> ").strip())
    context_ids = tokenizer.convert_tokens_to_ids(tokens, device=device)
    with torch.no_grad():
        top_ids, top_vals = model.get_closest_words_to_context(context_ids, target_pos)
        print("Results:")
        for idx, sim in zip(top_ids, top_vals):
            print(f"{tokenizer.vocab[idx]}\t{sim}")
