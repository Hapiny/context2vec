import os
import logging
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from time import time
from dataset.dataset import Dataset
from modeling.context2vec import Context2Vec

debug = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug else logging.WARNING)

# Create device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparams
lr = 1e-3
batch_size = 100
clip_norm = 5

# Create summary writer
log_dir = "./experiments/paper-params"
writer = SummaryWriter(log_dir)

# Load train word counter and all datasets
train_dataset_path = "./dataset/sample.txt"
# train_dataset_path = "./dataset/wikitext-103/wiki.train.tokens.processed"
valid_dataset_path = "./dataset/sample.txt"
# valid_dataset_path = "./dataset/wikitext-103/wiki.valid.tokens.processed"
test_dataset_path = "./dataset/sample.txt"
# test_dataset_path = "./dataset/wikitext-103/wiki.test.tokens.processed"

# Train dataset
logger.debug("Loading Train dataset...")
with open(train_dataset_path, "r") as fp:
    sentences = []
    for line in fp:
        tokens = line.lower().strip().split()
        num_tokens = len(tokens)
        if num_tokens > 64 or num_tokens < 2:
            continue
        sentences.append(tokens)
    sentences = np.array(sentences)

logger.debug("Create Train Dataset...")
train_dataset = Dataset(sentences, batch_size=batch_size, device=device, is_train=True)

model = Context2Vec(
    word_freqs=train_dataset.freqs,
    pad_token_id=train_dataset.pad_token_id,
    context_emb_size=300,
    target_emb_size=600,
    lstm_size=600,
    mlp_num_layers=2,
    mlp_size=600*2,
    mlp_dropout=0.0,
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
check_every = 300
step = 0
for epoch in range(1, num_epochs + 1):
    start_epoch = time()
    progress_bar = train_dataset.get_batch_iter()
    total_epoch_loss = 0.0
    for batch in progress_bar:
        optimizer.zero_grad()
        loss = model(batch.sentence)
        total_epoch_loss += loss.item()
        writer.add_scalar("Loss", scalar_value=loss, global_step=step)
        loss.backward()
        for name, param in model.named_parameters():
            writer.add_histogram(
                tag=f"Gradient norm of {name}",
                values=param.grad.norm(dim=-1),
                global_step=step
            )
        clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        step += 1

        if check_every > 0 and step and step % check_every == 0:
            writer.add_embedding(
                mat=model.target_embs.weight.data,
                metadata=train_dataset.vocab.itos,
                tag="Target embeddings",
                global_step=step
            )
            tokens = ["my dog is so cute and adorable !".lower().strip().split()]
            target_pos = 4
            context_ids = train_dataset.convert_tokens_to_ids(tokens)
            with torch.no_grad():
                top_ids, top_vals = model.get_closest_words_to_context(context_ids, target_pos)
                print("Results:")
                for idx, sim in zip(top_ids, top_vals):
                    print(f"{train_dataset.vocab.itos[idx]}\t{sim}")
    epoch_time = time() - start_epoch
    print(f"Epoch {epoch}. Total loss: {total_epoch_loss}")
    torch.save(model.state_dict(), os.path.join(log_dir, f"c2v_{epoch + 1}"))

writer.close()
