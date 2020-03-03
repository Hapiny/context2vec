import os

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import Dataset
from modeling.context2vec import Context2Vec

# Create device
device = torch.device("cuda:6")

# Hyperparams
lr = 1e-4
batch_size = 128
clip_norm = 5

# Create summary writer
log_dir = "./experiments/paper-params"
writer = SummaryWriter(log_dir)

# Load train word counter and all datasets
train_dataset_path = "./dataset/wikitext-103/wiki.train.tokens.processed"
valid_dataset_path = "./dataset/wikitext-103/wiki.valid.tokens.processed"
test_dataset_path = "./dataset/wikitext-103/wiki.test.tokens.processed"

# Create counter
counter_path = "./dataset/wikitext-103/wiki.train.tokens.counter"
counter = dict()
with open(counter_path, "r") as fp:
    for line in fp:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        word, count = parts
        counter[word] = int(count)

# Train dataset
print("Loading Train dataset...")
with open(train_dataset_path, "r") as fp:
    sentences = [line.strip().split() for line in fp.readlines()]
    sentences = np.array(sentences)

print("Create Train Dataset...")
train_dataset = Dataset(sentences, counter, batch_size=batch_size, device=device, is_train=True)

print("Create Valid Dataset...")
with open(valid_dataset_path, "r") as fp:
    sentences = [line.strip().split() for line in fp.readlines()]
    sentences = np.array(sentences)
valid_dataset = Dataset(sentences, vocab=train_dataset.vocab, batch_size=batch_size, device=device)

print("Create Test Dataset...")
with open(test_dataset_path, "r") as fp:
    sentences = [line.strip().split() for line in fp.readlines()]
    sentences = np.array(sentences)
test_dataset = Dataset(sentences, vocab=train_dataset.vocab, batch_size=batch_size, device=device)

model = Context2Vec(
    context_emb_size=300,
    target_emb_size=600,
    word_freqs=list(train_dataset.freqs.values()),
    lstm_size=600,
    mlp_num_layers=2,
    mlp_hidden_size=1200,
    mlp_dropout=0.1,
    alpha=0.75,
    num_negative_samples=10,
    device=device,
    summary_writer=writer
)

# Save model graph
seq_len = 26
input_ids = torch.randint(0, 50, (seq_len, batch_size), dtype=torch.long, device=device)
writer.add_graph(model, input_ids)

# Create optimizer
optimizer = Adam(model.parameters(), lr=lr)

num_epochs = 50
check_every = 1000
for epoch in range(num_epochs):
    progress_bar = train_dataset.get_batch_iter()
    batch_idx = 0
    for iterator in progress_bar:
        for batch in tqdm(iterator):
            optimizer.zero_grad()
            loss = model(batch.sentence)
            writer.add_scalar("Loss", scalar_value=loss, global_step=batch_idx)
            loss.backward()
            for name, param in model.named_parameters():
                writer.add_histogram(
                    tag=f"Gradient norm of {name}",
                    values=param.grad.norm(dim=-1),
                    global_step=batch_idx
                )
            clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            batch_idx += 1

            if check_every > 0 and batch_idx and batch_idx % check_every == 0:
                writer.add_embedding(
                    mat=model.target_embs.weight.data,
                    metadata=train_dataset.vocab.itos,
                    tag="Target embeddings",
                    global_step=batch_idx
                )
                tokens = ["my dog is so cute and adorable !".lower().strip().split()]
                target_pos = 4
                context_ids = train_dataset.convert_tokens_to_ids(tokens)
                with torch.no_grad():
                    top_ids, top_vals = model.get_closest_words_to_context(context_ids, target_pos)
                    print("Results:")
                    for idx, sim in zip(top_ids, top_vals):
                        print(f"{train_dataset.vocab.itos[idx]}\t{sim}")

    torch.save(model.state_dict(), os.path.join(log_dir, f"c2v_{epoch + 1}"))

writer.close()
