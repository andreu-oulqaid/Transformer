# src/data/embedding.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    """Dataset for paired sentences (e.g., English -> German)."""
    def __init__(self, dataset_split, tokenizer_src, tokenizer_tgt):
        self.data = dataset_split
        self.t_src = tokenizer_src
        self.t_tgt = tokenizer_tgt
        
        # Get special token IDs
        self.sos_id = self.t_tgt.token_to_id("[SOS]")
        self.eos_id = self.t_tgt.token_to_id("[EOS]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Encode Source (English)
        src_ids = self.t_src.encode(self.data[idx]['en']).ids
        
        # 2. Encode Target (German) and wrap with SOS/EOS
        tgt_ids = self.t_tgt.encode(self.data[idx]['de']).ids
        tgt_ids = [self.sos_id] + tgt_ids + [self.eos_id]

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long)
        }

class TextEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model,
            padding_idx=pad_id
        )

    def forward(self, input_ids):
        return self.embedding_layer(input_ids)

    def collate_fn(self, batch):
        """Pads both src and tgt in the batch."""
        src_list = [item['src'] for item in batch]
        tgt_list = [item['tgt'] for item in batch]

        src_padded = pad_sequence(src_list, batch_first=True, padding_value=self.pad_id)
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=self.pad_id)

        return src_padded, tgt_padded