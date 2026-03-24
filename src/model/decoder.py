# src/model/decoder.py

import torch.nn as nn
from src.model.attention import MultiHeadAttention
from src.model.feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Cross-attention (encoder output)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        # Feed Forward
        self.ffn = FeedForward(d_model, d_ff)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # -------------------------
        # Masked Self-Attention
        # -------------------------
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # -------------------------
        # Cross-Attention
        # -------------------------
        attn_output, weights = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # -------------------------
        # Feed Forward
        # -------------------------
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x, weights

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        last_weights = None
        for layer in self.layers:
            x, last_weights = layer(x, enc_output, tgt_mask, src_mask)
        return x, last_weights