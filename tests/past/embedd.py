# tests/test_embedding.py
import os
import torch
from src.data.embedding import TextEmbedder, TextDataset
from src.data.dataset import get_dataset
from src.data.tokenizer import get_tokenizer
from torch.utils.data import DataLoader

# Configuration
BATCH_SIZE = 32
D_MODEL = 256
CHECKPOINT_DIR = "checkpoints"
EMBEDDING_PATH = os.path.join(CHECKPOINT_DIR, "embedding_matrix_init.pt")
SAMPLE_BATCH_PATH = os.path.join(CHECKPOINT_DIR, "sample_batch_ids.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_test():
    # Load Data & Tokenizer
    dataset = get_dataset()
    tokenizer = get_tokenizer(dataset=dataset)
    pad_id = tokenizer.get_vocab()["[PAD]"]
    vocab_size = tokenizer.get_vocab_size()

    # Initialize Embedder
    embedder = TextEmbedder(vocab_size, D_MODEL, pad_id, DEVICE)
    
    # Handle Checkpoints
    if embedder.load_weights(EMBEDDING_PATH):
        print(f"Loaded embedding matrix from '{EMBEDDING_PATH}'")
    else:
        embedder.save_weights(EMBEDDING_PATH)
        print(f"Initialized and saved new embedding matrix.")

    # Prepare Data
    train_dataset = TextDataset(dataset['train']['en'], tokenizer)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=embedder.collate_fn
    )

    # Extract/Load Sample Batch
    if os.path.exists(SAMPLE_BATCH_PATH):
        sample_padded = torch.load(SAMPLE_BATCH_PATH).to(DEVICE)
        print(f"Loaded sample batch from '{SAMPLE_BATCH_PATH}'")
    else:
        sample_padded = next(iter(train_loader))
        torch.save(sample_padded.cpu(), SAMPLE_BATCH_PATH)
        print(f"Saved sample batch to '{SAMPLE_BATCH_PATH}'")

    #Embedding Lookup
    vectors = embedder.forward(sample_padded)
    print(f"Sample batch IDs shape: {sample_padded.shape}")  
    print(f"Embedded vectors shape: {vectors.shape}")       

    #  Sanity Check
    print("\n--- Weight Inspection ---")
    for word in ["dog", "cat", "[PAD]", "[UNK]"]:
        wid = tokenizer.get_vocab()[word]
        vector_sample = embedder.embedding_layer.weight[wid, :10]
        print(f"{word:7} (ID {wid:4}): {vector_sample.tolist()}")

if __name__ == "__main__":
    run_test()