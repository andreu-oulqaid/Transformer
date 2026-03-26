import torch
from tqdm import tqdm
from torchmetrics.text.bleu import BLEUScore
from src.model.transformer import Transformer
from src.data.dataset import get_dataset
from src.data.tokenizer import get_tokenizer
from src.inference.translator import Translator

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pt" # Or 
BEAM_SIZE = 5
D_MODEL = 256

def compute_corpus_bleu():
    print(f"Loading Evaluation Pipeline on {DEVICE}...")
    
    # Load Data & Tokenizers
    dataset = get_dataset("data/multi30k")
    test_data = dataset["test"]
    tokenizer_en = get_tokenizer("data/tokenizer_en.json", test_data, "en")
    tokenizer_de = get_tokenizer("data/tokenizer_de.json", test_data, "de")

    # Reconstruct Model
    model = Transformer(
        src_vocab_size=tokenizer_en.get_vocab_size(),
        tgt_vocab_size=tokenizer_de.get_vocab_size(),
        d_model=D_MODEL, num_heads=8, num_layers=3, d_ff=512
    ).to(DEVICE)

    # Load Weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    translator = Translator(model, tokenizer_en, tokenizer_de, DEVICE)
    bleu_metric = BLEUScore()

    predictions = []
    references = []

    print(f"Calculating BLEU for {len(test_data)} samples (Beam Size: {BEAM_SIZE})...")

    with torch.no_grad():
        for i in tqdm(range(len(test_data))):
            src_text = test_data[i]["en"]
            ref_text = test_data[i]["de"]

            # Translate using Beam Search Translator
            pred_text, _, _, _ = translator.translate(src_text, beam_size=BEAM_SIZE)

            predictions.append(pred_text)
            # BLEUScore expects a list of lists for references: [[ref1], [ref2]]
            references.append([ref_text])

    # Final Calculation
    score = bleu_metric(predictions, references)
    
    print("\n" + "="*30)
    print(f"FINAL CORPUS BLEU: {score.item():.4f}")
    print("="*30)

    # Save a few examples to a text file for manual inspection
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Corpus BLEU: {score.item():.4f}\n\n")
        for p, r in zip(predictions[:10], references[:10]):
            f.write(f"REF: {r[0]}\nPRED: {p}\n\n")

if __name__ == "__main__":
    compute_corpus_bleu()