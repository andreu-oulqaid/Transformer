import torch
import os
from src.model.transformer import Transformer
from src.data.tokenizer import get_tokenizer
from src.inference.translator import Translator
from src.evaluation.visualize import plot_attention, generate_interactive_html, export_attention_to_csv, export_attention_to_text

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/transformer_multi30k.pt"

# Ensure these match training settings exactly
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 3
D_FF = 512

def load_model_and_translator():
    # There has to be a checpoint model. If not, we can't run the demo.
    if not os.path.exists(CHECKPOINT_PATH):
        print(f" Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return None, None, None

    print(f" Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Load Tokenizers
    tokenizer_en = get_tokenizer("data/tokenizer_en.json", None, None)
    tokenizer_de = get_tokenizer("data/tokenizer_de.json", None, None)

    # Reconstruct Model Architecture
    model = Transformer(
        src_vocab_size=tokenizer_en.get_vocab_size(),
        tgt_vocab_size=tokenizer_de.get_vocab_size(),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF
    ).to(DEVICE)

    # Load Weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(" Model weights loaded successfully.")

    # Initialize Translator
    translator = Translator(model, tokenizer_en, tokenizer_de, DEVICE)
    
    return translator, tokenizer_en, tokenizer_de

def run_demo():
    translator, _, _ = load_model_and_translator()
    
    if translator is None:
        return

    # Test Sentences
    test_sentences = [
        "A man in an orange hat staring at something.",
        "A Boston Terrier is running on lush green grass.",
        "People are fixing the roof of a house."
    ]

    print("\n---  Transformer Demo ---")
    
    # In demo.py, update the loop:
    os.makedirs("figures/html", exist_ok=True)
    os.makedirs("figures/csv", exist_ok=True)

    for i, test_sent in enumerate(test_sentences):
        print(f"\nSRC: {test_sent}")
        
        translation, attn, src_toks, tgt_toks = translator.translate(test_sent, beam_size=5)
        print(f"PRED: {translation}")

        # Generate Interactive HTML for Human Analysis
        generate_interactive_html(attn, src_toks, tgt_toks, filename=f"figures/html/viz_sent_{i}.html")

        # Export matrix for AI Analysis (using Head 0 as example)
        matrix_string = export_attention_to_csv(attn, src_toks, tgt_toks, head_idx=0, filename=f"figures/csv/matrix_sent_{i}.csv")

        # --- NEW: Loop through all 8 heads ---
        os.makedirs(f"figures/plots/sentence_{i}", exist_ok=True)
        for head in range(NUM_HEADS):
            save_name = f"figures/plots/sentence_{i}/head_{head}.png"
            plot_attention(attn, src_toks, tgt_toks, head_idx=head, save_path=save_name)
        
        # --- NEW: Export weights for AI analysis ---
        export_attention_to_text(attn, src_toks, tgt_toks, head_idx=0, 
                                 save_path=f"figures/plots/sentence_{i}/weights_head0.txt")

    print("\n Demo complete! Check the .png files in your directory to see word alignments.")

if __name__ == "__main__":
    run_demo()