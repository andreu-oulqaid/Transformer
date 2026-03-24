# src/model/positional_encoding.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Positions: [0, 1, 2, ..., max_len]
        position = torch.arange(0, max_len).unsqueeze(1)

        # Compute scaling factor (avoiding power)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension → [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (NOT trainable)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]

        return x