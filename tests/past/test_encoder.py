# tests/test_encoder.py
import torch
from torch.utils.data import DataLoader

# Import your data logic
from src.data.dataset import get_dataset
from src.data.tokenizer import get_tokenizer
from src.data.embedding import TextEmbedder, TextDataset

# Import your model logic
from src.model.encoder import Encoder

def test_full_encoder_pipeline():
    # --- 1. CONFIGURATION ---
    D_MODEL = 256
    NUM_HEADS = 8
    D_FF = 512
    NUM_LAYERS = 3
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. DATA LOAD ---
    print("Initializing Data Pipeline...")
    ds = get_dataset()
    tokenizer = get_tokenizer(dataset=ds)
    
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    # Create actual PyTorch objects
    train_ds = TextDataset(ds['train']['en'][:100], tokenizer) # Test with first 100 samples
    
    embedder = TextEmbedder(vocab_size, D_MODEL, pad_id, DEVICE)
    
    loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=embedder.collate_fn
    )

    # --- 3. MODEL LOAD ---
    print("Initializing Encoder...")
    encoder = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF).to(DEVICE)

    # --- 4. EXECUTION ---
    # Get one real batch
    batch_ids = next(iter(loader)).to(DEVICE)
    
    print(f"Batch IDs shape: {batch_ids.shape}") # [Batch, Seq_Len]
    
    # Pass through Embedder
    embeddings = embedder(batch_ids)
    print(f"Embeddings shape: {embeddings.shape}") # [Batch, Seq_Len, D_MODEL]

    # Pass through Encoder
    encoded_output = encoder(embeddings)
    print(f"Encoder output shape: {encoded_output.shape}")

    # --- 5. VERIFICATION ---
    assert encoded_output.shape == embeddings.shape
    print("\nSuccess! The Encoder processed a real Multi30k batch correctly.")

if __name__ == "__main__":
    test_full_encoder_pipeline()