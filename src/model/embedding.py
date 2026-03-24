# src/data/embedding.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.data.dataset import get_dataset
from src.data.tokenizer import get_tokenizer

# ----------------------------
# PARAMETERS
# ----------------------------
BATCH_SIZE = 32
D_MODEL = 256
CHECKPOINT_DIR = "checkpoints"
EMBEDDING_PATH = os.path.join(CHECKPOINT_DIR, "embedding_matrix_init.pt")
SAMPLE_BATCH_PATH = os.path.join(CHECKPOINT_DIR, "sample_batch_ids.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD DATA AND TOKENIZER
# ----------------------------
dataset = get_dataset()
tokenizer = get_tokenizer(dataset=dataset)
PAD_ID = tokenizer.get_vocab()["[PAD]"]
VOCAB_SIZE = tokenizer.get_vocab_size()

# ----------------------------
# DEFINE EMBEDDING LAYER
# ----------------------------
embedding_layer = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=D_MODEL).to(DEVICE)

# ----------------------------
# CUSTOM DATASET CLASS
# ----------------------------
class TextDataset(Dataset):
    """PyTorch Dataset wrapping Multi30k English sentences"""
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Returns list of token IDs
        return torch.tensor(tokenizer.encode(self.sentences[idx]).ids, dtype=torch.long)

# ----------------------------
# COLLATE FUNCTION (for dynamic padding)
# ----------------------------
def collate_fn(batch):
    # batch: list of tensors (token IDs)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    return padded_batch.to(DEVICE)

# ----------------------------
# CREATE DATALOADER
# ----------------------------
train_dataset = TextDataset(dataset['train']['en'])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ----------------------------
# CHECKPOINT DIRECTORY
# ----------------------------
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------
# LOAD EMBEDDING CHECKPOINT IF EXISTS
# ----------------------------
if os.path.exists(EMBEDDING_PATH):
    embedding_layer.load_state_dict(torch.load(EMBEDDING_PATH, map_location=DEVICE))
    print(f"Loaded embedding matrix from '{EMBEDDING_PATH}'")

# ----------------------------
# EXTRACT ONE SAMPLE BATCH FOR INSPECTION
# ----------------------------
if os.path.exists(SAMPLE_BATCH_PATH):
    sample_padded = torch.load(SAMPLE_BATCH_PATH).to(DEVICE)
    print(f"Loaded sample batch from '{SAMPLE_BATCH_PATH}'")
else:
    # Take first batch from DataLoader
    sample_padded = next(iter(train_loader))
    torch.save(sample_padded.cpu(), SAMPLE_BATCH_PATH)
    print(f"Saved sample batch to '{SAMPLE_BATCH_PATH}'")

# ----------------------------
# EMBEDDING LOOKUP
# ----------------------------
vectors = embedding_layer(sample_padded)
print(f"Sample batch padded IDs shape: {sample_padded.shape}")     # [batch_size, seq_len]
print(f"Embedded vectors shape: {vectors.shape}")                  # [batch_size, seq_len, D_MODEL]

# ----------------------------
# SAVE EMBEDDING MATRIX (INITIAL)
# ----------------------------
if not os.path.exists(EMBEDDING_PATH):
    torch.save(embedding_layer.state_dict(), EMBEDDING_PATH)
    print(f"Saved embedding matrix to '{EMBEDDING_PATH}'")

# ----------------------------
# SANITY CHECK
# ----------------------------
for word in ["dog", "cat", "[PAD]", "[UNK]"]:
    wid = tokenizer.get_vocab()[word]
    vector_sample = embedding_layer.weight[wid, :10]
    print(f"{word} (ID {wid}): {vector_sample.tolist()}")