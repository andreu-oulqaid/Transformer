import os
import torch
import torch.nn as nn
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from src.data.dataset import get_dataset
from src.data.tokenizer import get_tokenizer
from src.inference.translator import Translator
from src.model.transformer import Transformer
from src.model.masks import create_padding_mask, create_decoder_mask
from src.evaluation.evaluate import evaluate
from src.evaluation.visualize import plot_loss # Ensure this exists!

# -------------------------
# CONFIGURATION
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/transformer_multi30k.pt"
BEST_MODEL_PATH = "checkpoints/best_model.pt"
D_MODEL = 256
BATCH_SIZE = 32
EPOCHS = 150
WARMUP_STEPS = 4000
PATIENCE = 15  # Stop if no improvement for 15 epochs

def lr_lambda(current_step: int):
    step = max(1, current_step)
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

def collate_fn(batch, tokenizer_src, tokenizer_tgt):
    pad_id = tokenizer_src.token_to_id("[PAD]")
    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    eos_id = tokenizer_tgt.token_to_id("[EOS]")
    src_ids, tgt_ids = [], []
    for item in batch:
        src_ids.append(torch.tensor(tokenizer_src.encode(item["en"]).ids))
        t_ids = [sos_id] + tokenizer_tgt.encode(item["de"]).ids + [eos_id]
        tgt_ids.append(torch.tensor(t_ids))
    return (nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id),
            nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id))

def main():
    start_time = time.time()
    print(f" Training started at: {datetime.now().strftime('%H:%M:%S')}")
    
    dataset = get_dataset("data/multi30k")
    tokenizer_en = get_tokenizer("data/tokenizer_en.json", dataset, "en")
    tokenizer_de = get_tokenizer("data/tokenizer_de.json", dataset, "de")
    pad_id = tokenizer_en.token_to_id("[PAD]")

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer_en, tokenizer_de))
    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE,
                            collate_fn=lambda b: collate_fn(b, tokenizer_en, tokenizer_de))

    model = Transformer(
        src_vocab_size=tokenizer_en.get_vocab_size(),
        tgt_vocab_size=tokenizer_de.get_vocab_size(),
        d_model=D_MODEL, num_heads=8, num_layers=3, d_ff=512, dropout=0.1
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1) # Ignore padding and correct word 90%, 10% other words
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9) # Params from Attention is All You Need paper
    scheduler = LambdaLR(optimizer, lr_lambda)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {"train": [], "val": []}

    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0
            
            for i, (src, tgt) in enumerate(train_loader):
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                src_mask = create_padding_mask(src, pad_id)
                tgt_mask = create_decoder_mask(tgt_input, pad_id)

                preds, _ = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step() # Every batch
                total_train_loss += loss.item()

            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                    tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                    src_mask = create_padding_mask(src, pad_id)
                    tgt_mask = create_decoder_mask(tgt_input, pad_id)
                    preds, _ = model(src, tgt_input, src_mask, tgt_mask)
                    v_loss = criterion(preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1))
                    total_val_loss += v_loss.item()
            
            avg_train = total_train_loss / len(train_loader)
            avg_val = total_val_loss / len(val_loader)
            history["train"].append(avg_train)
            history["val"].append(avg_val)
            
            print(f"Ep {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Time: {time.time()-epoch_start:.1f}s")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                torch.save({'model_state_dict': model.state_dict()}, BEST_MODEL_PATH)
                print(f"New Best Model Saved!")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving state...")

    # Final Save
    torch.save({'model_state_dict': model.state_dict(), 'd_model': D_MODEL}, CHECKPOINT_PATH)
    
    total_duration = (time.time() - start_time) / 60
    print(f"Finished in {total_duration:.2f} mins. Best Val Loss: {best_val_loss:.4f}")

    # Final Demo Evaluation
    model.load_state_dict(torch.load(BEST_MODEL_PATH)['model_state_dict'])
    model.eval()
    translator = Translator(model, tokenizer_en, tokenizer_de, DEVICE)
    sentence = "A man in an orange hat."
    translation, attn, src_toks, tgt_toks = translator.translate(sentence)
    print(f"Sample Result: {translation}")

if __name__ == "__main__":
    main()