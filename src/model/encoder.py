# src/model/encoder.py

import torch.nn as nn
from src.model.attention import MultiHeadAttention
from src.model.feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # -------------------------
        # Multi-Head Attention
        # -------------------------
        attn_output, weights = self.attention(x, x, x, mask)

        # Residual + Norm
        x = self.norm1(x + self.dropout(attn_output))

        # -------------------------
        # Feed Forward
        # -------------------------
        ffn_output = self.ffn(x)

        # Residual + Norm
        x = self.norm2(x + self.dropout(ffn_output))

        return x, weights
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        last_weights = None
        for layer in self.layers:
            x, last_weights = layer(x, mask)
        return x, last_weights