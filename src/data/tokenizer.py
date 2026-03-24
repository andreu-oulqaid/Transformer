
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_tokenizer(tokenizer_path, dataset, lang_key):
    """
    lang_key: 'en' or 'de'
    """
    if os.path.exists(tokenizer_path):
        return Tokenizer.from_file(tokenizer_path)

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=2
    )

    # Train only on the specific language column
    def get_sentences():
        for item in dataset['train']:
            yield item[lang_key]

    tokenizer.train_from_iterator(get_sentences(), trainer=trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer