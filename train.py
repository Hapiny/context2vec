import logging
import os
import warnings
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import Dataset
from modeling.context2vec import Context2Vec
from modeling.negative_sampling import NegativeSampling

warnings.filterwarnings("ignore")

debug = True
logger = logging.getLogger(__name__)
if debug:
    logger.setLevel(logging.INFO)

# Create device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparams
lr = 1e-3
batch_size = 256
clip_norm = 5

# Create summary writer
log_dir = "./experiments/parallel"
writer = SummaryWriter(log_dir)

# Load train word counter and all datasets
# train_dataset_path = "./dataset/sample.txt"
train_dataset_path = "./dataset/wikitext-103/wiki.train.tokens.processed"

# Train dataset
logger.info("Loading Train dataset...")
with open(train_dataset_path, "r") as fp:
    sentences = []
    for line in fp:
        tokens = line.lower().strip().split()
        num_tokens = len(tokens)
        if num_tokens > 64 or num_tokens < 2:
            continue
        sentences.append(tokens)
    sentences = np.array(sentences)

logger.info("Create Train Dataset...")
train_dataset = Dataset(sentences, batch_size=batch_size, device=device, is_train=True)

model = Context2Vec(
    pad_token_id=train_dataset.pad_token_id,
    vocab_size=len(train_dataset.vocab),
    encoder_type="transformer",
    context_emb_size=1024,
    target_emb_size=786,
    lstm_size=512,
    mlp_num_layers=2,
    mlp_size=600 * 2,
    mlp_dropout=0.0,
    summary_writer=writer
)
model.to(device)
print(model)
criterion = NegativeSampling(
    embeddings=model.target_embs,
    freqs=train_dataset.freqs,
    alpha=0.75,
    num_negative_samples=10
)

multi_gpu = False
if torch.cuda.device_count() > 1:
    multi_gpu = True
    parallel_model = nn.DataParallel(model, device_ids=[0, 1, 2]).to(device)

# Create optimizer
optimizer = Adam(model.parameters(), lr=lr)

num_epochs = 50
check_every = 1000
step = 0
for epoch in range(1, num_epochs + 1):
    start_epoch = time()
    progress_bar = train_dataset.get_batch_iter()
    total_loss = 0.0
    for batch in progress_bar:
        optimizer.zero_grad()

        if multi_gpu:
            context_vectors, target_vectors = parallel_model(batch.sentence)
        else:
            context_vectors, target_vectors = model(batch.sentence)
        loss = criterion(target_vectors, context_vectors)

        total_loss += loss.item()
        writer.add_scalar("Loss", scalar_value=loss, global_step=step)
        loss.backward()
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
    epoch_time = time() - start_epoch
    print(f"Epoch {epoch} ({epoch_time} sec.)")
    torch.save(model.state_dict(), os.path.join(log_dir, f"c2v_{epoch}"))
