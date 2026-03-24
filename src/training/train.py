import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from src.data.dataset import get_dataset
from src.data.tokenizer import get_tokenizer
from src.inference.translator import Translator
from src.model.transformer import Transformer
from src.model.masks import create_padding_mask, create_decoder_mask
from src.evaluation.evaluate import evaluate

# -------------------------
# CONFIGURATION & DEVICE
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/transformer_multi30k.pt"
D_MODEL = 256
BATCH_SIZE = 32
EPOCHS = 15
WARMUP_STEPS = 4000

def lr_lambda(current_step: int):
    """Transformer LR schedule: Linear warmup then inverse square root decay."""
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
    print(" Starting training pipeline...")
    if torch.cuda.is_available():
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")

    # LOAD DATA & TOKENIZERS
    dataset = get_dataset("data/multi30k")
    tokenizer_en = get_tokenizer("data/tokenizer_en.json", dataset, "en")
    tokenizer_de = get_tokenizer("data/tokenizer_de.json", dataset, "de")
    pad_id = tokenizer_en.token_to_id("[PAD]")

    train_loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer_en, tokenizer_de)
    )

    # MODEL INITIALIZATION
    model = Transformer(
        src_vocab_size=tokenizer_en.get_vocab_size(),
        tgt_vocab_size=tokenizer_de.get_vocab_size(),
        d_model=D_MODEL, num_heads=8, num_layers=3, d_ff=512
    ).to(DEVICE)

    # LOSS, OPTIMIZER, SCHEDULER
    # Label smoothing prevents overconfidence and improves BLEU
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # TRAINING LOOP
    print(f" Training on {DEVICE}...")
    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            
            for i, (src, tgt) in enumerate(train_loader):
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

                src_mask = create_padding_mask(src, pad_id)
                tgt_mask = create_decoder_mask(tgt_input, pad_id)

                # Forward
                preds, _ = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1))

                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent exploding values
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                if i % 200 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Ep {epoch} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} Complete. Average Loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\n Training interrupted by user. Saving current state...")

    # SAVE MODEL
    print(f"Saving model to {CHECKPOINT_PATH}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer_en_path': "data/tokenizer_en.json",
        'tokenizer_de_path': "data/tokenizer_de.json",
        'd_model': D_MODEL
    }, CHECKPOINT_PATH)

    # FINAL EVALUATION
    print("\nRunning final evaluation...")
    model.eval()
    evaluate(model, dataset, tokenizer_en, tokenizer_de, DEVICE, num_samples=10)

    translator = Translator(model, tokenizer_en, tokenizer_de, DEVICE)
    sentence = "A man in an orange hat."

    # Get results
    translation, attn, src_tokens, tgt_tokens = translator.translate(sentence)

    print(f"Result: {translation}")

    # Plot it!
    from src.evaluation.visualize import plot_attention
    plot_attention(attn, src_tokens, tgt_tokens, head_idx=2) # we can visualize any head.
    
if __name__ == "__main__":
    main()