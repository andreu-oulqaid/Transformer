from src.data.dataset import get_dataset
from src.data.tokenizer import get_tokenizer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import os

# =========================
# LOAD DATA
# =========================

dataset = get_dataset()
# dataset['train'] → list-like structure
# Each item: {'en': sentence (str), 'de': sentence (str)}

tokenizer = get_tokenizer(dataset=dataset)

vocab = tokenizer.get_vocab()
# vocab: dict → {word: id}

# =========================
# SETUP PARAMETERS
# =========================

vocab_size = tokenizer.get_vocab_size()
d_model = 256  # embedding dimension

# =========================
# DEFINE EMBEDDING LAYER
# =========================

embedding_layer = nn.Embedding(
    num_embeddings=vocab_size,  # number of rows
    embedding_dim=d_model       # number of columns
)

# Internally:
# embedding_layerr.weight.shape = [vocab_size, 256]
# Each row = vector for one word

# =========================
# BATCHING SETUP
# =========================

batch_size = 32
pad_id = vocab["[PAD]"]

all_embeddings = []  # will store outputs per batch
# =========================
# CHECK IF EMBEDDINGS ALREADY SAVED
# =========================

checkpoint_matrix_path = "checkpoints/embedding_matrix_init.pt"
checkpoint_sample_path = "checkpoints/sample_batch_ids.pt"

# =========================
# LOAD EMBEDDING FROM CHECKPOINT IF AVAILABLE
# =========================

if os.path.exists(checkpoint_matrix_path):
    embedding_layer.load_state_dict(torch.load(checkpoint_matrix_path))
    print(f"Loaded embedding matrix from '{checkpoint_matrix_path}'")
save = False
if os.path.exists(checkpoint_sample_path):
    sample_padded = torch.load(checkpoint_sample_path)
    print(f"Loaded sample batch from '{checkpoint_sample_path}'")
# Only run the loop if checkpoints do not exist

if not (os.path.exists(checkpoint_matrix_path) and os.path.exists(checkpoint_sample_path)):

    # =========================
    # MAIN LOOP (BATCH PROCESSING)
    # =========================
    save = True
    for i in range(0, len(dataset['train']), batch_size):

        # -------------------------
        # 1. SELECT BATCH OF SENTENCES
        # -------------------------
        start = i
        end = i + batch_size

        # -------------------------
        # 2. EXTRACT SENTENCES
        # -------------------------
        sentences = dataset['train']['en'][start:end]
        # sentences: list of strings (batch_size sentences)

        # -------------------------
        # 3. TOKENIZE
        # -------------------------
        encodings = [tokenizer.encode(s).ids for s in sentences]

        # -------------------------
        # 4. CONVERT TO TENSORS
        # -------------------------
        input_ids = [torch.tensor(ids) for ids in encodings]

        # -------------------------
        # 5. PAD SEQUENCES
        # -------------------------
        padded = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_id
        )

        # -------------------------
        # 6. EMBEDDING LOOKUP
        # -------------------------
        vectors = embedding_layer(padded)

        # vectors.shape = [batch_size, max_seq_len, 256]

        # -------------------------
        # 7. STORE (optional)
        # -------------------------
        all_embeddings.append(vectors)
    # =========================
    # DEBUG / INSPECTION
    # =========================

    print(f"Padded input shape: {padded.shape}")           # [batch_size, seq_len]
    print(f"Embedded tensor shape: {vectors.shape}")      # [batch_size, seq_len, 256]

    # Inspect first word of first sentence
    print("\nFirst token vector (first 10 values):")
    print(vectors[0, 0, :10])  # shape [256]


# =========================
# SANITY CHECKS / SAVE INITIAL EMBEDDINGS
# =========================

os.makedirs("checkpoints", exist_ok=True)

# 1. Save initial embedding matrix
if save:
    torch.save(embedding_layer.state_dict(), "checkpoints/embedding_matrix_init.pt")
    print("Saved initial embedding matrix to 'checkpoints/embedding_matrix_init.pt'")

# 2. Inspect a few word vectors
print("\n--- SANITY CHECK: SAMPLE VECTORS ---")
for word in ["dog", "cat", "[PAD]", "[UNK]"]:
    word_id = vocab[word]
    vector_sample = embedding_layer.weight[word_id, :10]  # first 10 dimensions
    print(f"Word: '{word}', ID: {word_id}, vector[:10]: {vector_sample.tolist()}")

# 3. Save a small sample batch of padded IDs for quick testing
sample_batch = dataset['train']['en'][:batch_size]
sample_sentences = [item for item in sample_batch]
sample_encodings = [tokenizer.encode(s).ids for s in sample_sentences]
sample_input_ids = [torch.tensor(ids) for ids in sample_encodings]
sample_padded = pad_sequence(sample_input_ids, batch_first=True, padding_value=pad_id)
if save:
    torch.save(sample_padded, "checkpoints/sample_batch_ids.pt")
    print("Saved a small padded batch of IDs to 'checkpoints/sample_batch_ids.pt'")

# Show shapes for sanity
print("\nSanity check complete. Shapes:")
#print(f"Dataset embeddings shape: {all_embeddings.shape}")  
print(f"Embedding matrix: {embedding_layer.weight.shape}")  # [vocab_size, d_model]
print(f"Sample batch padded IDs: {sample_padded.shape}")    # [batch_size, max_seq_len]