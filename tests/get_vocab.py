import os
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# =========================
# PATHS
# =========================
DATA_PATH = "../data/multi30k"
TOKENIZER_PATH = "../data/tokenizer_en.json"

# =========================
# LOAD DATASET
# =========================
print("\n--- STEP 1: LOADING DATASET ---")

if os.path.exists(DATA_PATH) and len(os.listdir(DATA_PATH)) > 0:
    print("Loading dataset from disk...")
    dataset = load_from_disk(DATA_PATH)
else:
    print("Downloading dataset...")
    dataset = load_dataset("bentrevett/multi30k")
    dataset.save_to_disk(DATA_PATH)

# Inspect dataset
train_dataset = dataset['train']
first_pair = train_dataset[0]
print(f"Total sentences in training: {len(train_dataset)}")
print(f"Raw Entry Example: {first_pair}")
print(f"English: {first_pair['en']}")
print(f"German: {first_pair['de']}")

# =========================
# TOKENIZER
# =========================
print("\n--- STEP 2: TOKENIZER ---")
def get_tokenizer():
    if os.path.exists(TOKENIZER_PATH):
        print("Loading tokenizer from disk...")
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        return tokenizer
    else:
        print("Training tokenizer...")

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        def get_english_sentences():
            for item in dataset['train']:
                yield item['en']

        tokenizer.train_from_iterator(get_english_sentences(), trainer=trainer)

        # Save tokenizer
        tokenizer.save(TOKENIZER_PATH)
        print(f"Tokenizer saved to {TOKENIZER_PATH}")
    return tokenizer

tokenizer = get_tokenizer()

# =========================
# VOCABULARY
# =========================
print("\n--- STEP 3: VOCABULARY ---")

vocab = tokenizer.get_vocab()
print(f"Vocabulary size: {len(vocab)}")

word = "dog"
print(f"ID for '{word}': {vocab.get(word)}")
print(f"ID for [PAD]: {vocab.get('[PAD]')}")

# =========================
# ENCODING
# =========================
print("\n--- STEP 4: ENCODING ---")

sentence = "Two dogs are running."
encoded = tokenizer.encode(sentence)

print(f"Original: {sentence}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")