import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_models.config import *
from mamba_models.layers import *
from mamba_models.attention.encoder import Encoder

class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(BidirectionalMambaBlock, self).__init__()
        self.feature_dim = d_ff

        self.fmamba = Mamba(d_model, d_state=MAMBA_D_STATE, d_conv=MAMBA_D_CONV, expand=MAMBA_EXPAND)
        self.bmamba = Mamba(d_model, d_state=MAMBA_D_STATE, d_conv=MAMBA_D_CONV, expand=MAMBA_EXPAND)

        self.an1 = AddNorm(d_model, dropout)
        self.an2 = AddNorm(d_model, dropout)
        self.an3 = AddNorm(d_model, dropout)

        self.feedforward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        B, W, C, D = x.shape  # Expected input shape: (B, W, C, D)

        # Ensure correct feature dimension for Mamba
        flattened_dim = C * D
        x = x.view(B, W, flattened_dim)  # Shape: (B, W, C * D)
        forward_dir = x
        backward_dir = torch.flip(x, dims=[1])  # Flip along sequence length
        

        forward_dir = self.fmamba(forward_dir)  # Pass flattened input to Mamba
        backward_dir = self.bmamba(backward_dir)

        backward_dir = torch.flip(backward_dir, dims=[1])  # Flip back

        # Add and normalize
        forward_dir = self.an1(x, forward_dir)
        backward_dir = self.an2(x, backward_dir)

        out = forward_dir + backward_dir
        out = self.feedforward(out)
        out = self.an3(out, out)

        # Reshape back to (B, W, C, D)
        out = out.view(B, W, C, D)
        return out



class BidirectionalMamba(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(BidirectionalMamba, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model * num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model * num_heads)  # Adjusted to match new feature size

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, W)
        x = self.encoder(input_ids, attention_mask)  # Output shape: (B, W, C, D)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x.view(x.shape[0], x.shape[1], -1))  # Apply normalization after flattening

        return x.view(x.shape[0], x.shape[1], x.shape[2], -1)  # Reshape back to (B, W, C, D)
