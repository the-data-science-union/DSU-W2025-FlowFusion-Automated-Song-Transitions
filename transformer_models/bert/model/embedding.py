import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = torch.zeros(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids=None):

        return self.token_embed(input_ids)
