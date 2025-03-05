import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & Norm
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout=0.1, device='cuda'):
        super(Encoder, self).__init__()
        self.device = device
        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, mask):
        # Input x is precomputed embeddings (e.g., [3, 1600, 4, 256])
        x = x.to(self.device)  # Ensure input is on the correct device
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x  # Output shape should match input shape (e.g., [3, 1600, 4, 256])