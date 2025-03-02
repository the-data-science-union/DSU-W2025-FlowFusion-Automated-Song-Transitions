import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(num_heads * self.d_k, self.d_model)  # Fix dimensions
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Reshape mask to match attn_scores dimensions
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape becomes (32, 1, 1, 15)
            mask = mask.expand_as(attn_scores)     # Expand to (32, 12, 15, 15)
            
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        return output, attn_probs
        
    def split_heads(self, x):
        """Split input into multiple attention heads for each channel independently"""
        batch_size, seq_length, num_channels, d_model = x.shape

        # Split the input into multiple heads along the d_model dimension
        head_dim = d_model // self.num_heads  # Ensure d_model is divisible by num_heads
        assert d_model % self.num_heads == 0, f"d_model must be divisible by num_heads"

        x = x.view(batch_size, seq_length, num_channels, self.num_heads, head_dim)  # (B, S, C, H, D_head)
        x = x.permute(0, 3, 1, 2, 4)  # (B, H, S, C, D_head)
        return x


    def combine_heads(self, x):
        batch_size, num_heads, seq_length, num_channels, d_k = x.size()
        x = x.permute(0, 2, 3, 1, 4).contiguous()  # (B, S, C, H, D_head)
        x = x.view(batch_size, seq_length, num_channels, num_heads * d_k)  # (B, S, C, D)
        return x

        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        combined_output = self.combine_heads(attn_output)
        output = self.W_o(combined_output)

        return output

