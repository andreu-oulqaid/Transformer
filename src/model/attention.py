# src/model/attention.py

import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch, heads, seq_len, d_k]
        """

        # Compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # shape: [batch, heads, seq_len, seq_len]

        # Scale
        d_k = Q.size(-1)
        scores = scores / math.sqrt(d_k)

        # Mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        out, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.d_model)

        # Final projection
        out = self.fc_out(out)

        return out, attn_weights