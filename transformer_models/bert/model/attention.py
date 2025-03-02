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
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if 1==2:
            # Expand the mask to match the shape of the attention scores
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, seq_length]
            mask = mask.expand(-1, self.num_heads, -1, -1)  # Shape: [batch_size, num_heads, seq_length, seq_length]

            # Apply the mask (set masked positions to -infinity)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Compute attention output
        output = torch.matmul(attn_probs, V)

        return output, attn_probs

        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output[0]))
        return output
