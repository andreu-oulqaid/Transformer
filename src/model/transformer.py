# src/model/transformer.py

import torch
import torch.nn as nn

from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Encoder & Decoder
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)

        # Final linear layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [batch, src_seq_len]
        tgt: [batch, tgt_seq_len]
        """

        # -------------------------
        # Encoder
        # -------------------------
        src_emb = self.dropout(self.pos_encoder(self.src_embedding(src)))
        enc_output, enc_weights = self.encoder(src_emb, src_mask)

        # -------------------------
        # Decoder
        # -------------------------
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embedding(tgt)))
        dec_output, dec_weights = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)

        # -------------------------
        # Output projection
        # -------------------------
        output = self.fc_out(dec_output)

        return output, dec_weights