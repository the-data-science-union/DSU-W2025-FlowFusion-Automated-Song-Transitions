import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        if (d_model % num_heads):
            print("d_model and num_heads: ", d_model, num_heads)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(num_heads * self.d_k, self.d_model)  # Fix dimensions
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Ensure mask is broadcastable to [batch_size, num_heads, seq_len, seq_len]
            if len(mask.shape) == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                mask = mask.expand(-1, attn_scores.shape[1], attn_scores.shape[2], -1)  # [batch_size, num_heads, seq_len, seq_len]
            elif len(mask.shape) == 3:  # [batch_size, 1, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, 1, seq_len]
                mask = mask.expand(-1, attn_scores.shape[1], attn_scores.shape[2], -1)
        
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        return output, attn_probs
        
    def split_heads(self, x):
        """Split input into multiple attention heads for each channel independently"""
        batch_size, seq_length, num_channels = x.shape

        # Ensure num_channels is divisible by num_heads
        head_dim = num_channels // self.num_heads
        assert num_channels % self.num_heads == 0, "num_channels must be divisible by num_heads"
        x = x.view(batch_size, seq_length, self.num_heads, head_dim)  # (B, S, H, D_head)

        x = x.permute(0, 2, 1, 3)  # (B, H, S, D_head)

        return x


    def combine_heads(self, x):
        """Combine multiple attention heads back into a single tensor."""

        batch_size, num_heads, seq_length, head_dim = x.shape #should be B, 4, seqL, #

        x = x.permute(0, 2, 1, 3).contiguous()  # (B, S, H, D_head)
        x = x.view(batch_size, seq_length, num_heads * head_dim)  # (B, S, D)

        return x
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        combined_output = self.combine_heads(attn_output)
        output = self.W_o(combined_output)

        return output

