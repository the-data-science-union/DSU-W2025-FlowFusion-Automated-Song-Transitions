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
        
        self.bn1 = nn.BatchNorm1d(d_model)  # BatchNorm after fmamba
        self.bn2 = nn.BatchNorm1d(d_model)  # BatchNorm after bmamba
        self.bn3 = nn.BatchNorm1d(d_model)  # BatchNorm after feedforward

    def forward(self, x):
        forward_dir = x
        backward_dir = torch.flip(x, dims=[1])  # Flip along sequence length
        
        forward_dir = self.fmamba(forward_dir)  # Pass flattened input to Mamba
        forward_dir = self.bn1(forward_dir.transpose(1, 2)).transpose(1, 2)  # Apply BatchNorm

        backward_dir = self.bmamba(backward_dir)
        backward_dir = self.bn2(backward_dir.transpose(1, 2)).transpose(1, 2)  # Apply BatchNorm

        backward_dir = torch.flip(backward_dir, dims=[1])  # Flip back

        # Add and normalize
        forward_dir = self.an1(x, forward_dir)
        backward_dir = self.an2(x, backward_dir)

        out = forward_dir + backward_dir
        out = self.feedforward(out)
        out = self.bn3(out.transpose(1, 2)).transpose(1, 2)  # Apply BatchNorm
        out += x

        return out



class BidirectionalMamba(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1, device="cuda"):
        super(BidirectionalMamba, self).__init__()
        self.d_model=d_model
        self.device = device
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout, device=self.device)
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model, d_ff, dropout) for _ in range(num_layers)
        ])
        self.final = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        B, S, C = input_ids.shape
        x = self.encoder(input_ids, attention_mask)
        for layer in self.layers:
            x = layer(x)
        x = x.view(B, S, C, self.d_model)
        x = self.final(x)

        return x
