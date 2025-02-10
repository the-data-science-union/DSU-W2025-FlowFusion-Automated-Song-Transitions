import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(max_seq_length, d_model)
        self.segment_embed = nn.Embedding(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        # Ensure input_ids and segment_ids are Long tensors
        input_ids = input_ids.long() # Ensure these are integers
        segment_ids = segment_ids.long() # Ensure these are integers

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeddings = self.token_embed(input_ids)
        position_embeddings = self.position_embed(position_ids)
        segment_embeddings = self.segment_embed(segment_ids)

        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
