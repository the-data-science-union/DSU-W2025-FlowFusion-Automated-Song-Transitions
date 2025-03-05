import torch
import torch.nn as nn
from .attention import MultiHeadAttention
import math

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
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & Norm
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1, device='cuda'):
        super(Encoder, self).__init__()
        self.device = device  # Store device
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)  # Embedding layer for vocab (optional)
        self.pos_encoding = self.positional_encoding(max_seq_length, d_model).to(device)  # Move positional encoding to device
        self.flatten = nn.Flatten(start_dim=-2)  # Flatten the input across the channel dimension
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def positional_encoding(self, max_seq_length, d_model):
        pos_encoding = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # Shape: [1, max_seq_length, d_model]

    def forward(self, x, mask):
        # x is the input tensor of shape [batch_size, seq_length, channels]
        
        # Apply flattening to collapse the channel dimension (4 channels → one flattened vector)
        x = self.flatten(x).to(self.device)  # Move x to the device

        # Add positional encoding (Broadcast pos across batch)
        x = self.embed(x.long()).to(self.device)  # Ensure embedding is on the same device
        pos = self.pos_encoding[:, :x.size(1)].to(self.device)  # Get positional encoding for the current sequence length
        pos = pos.repeat(1, 4, 1).to(self.device)  # Ensure positional encoding is on the same device
        x += pos  # Shape: [batch_size, seq_length, channels * d_model]
        
        x = self.dropout(x)

        # Pass through the layers of EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
