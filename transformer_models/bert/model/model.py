import math
import torch
import torch.nn as nn
from .encoder import Encoder
from .bert_heads import BERTPreTrainingHeads
from .. import config

class BERT_model(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(BERT_model, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.pos_encoding = self.positional_encoding(max_seq_length, d_model)
        
        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)  # Removed device param
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()
        self.pre_training_heads = BERTPreTrainingHeads(d_model, vocab_size)
        self.final = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)

    def positional_encoding(self, max_seq_length, d_model):
        pos_encoding = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # [1, max_seq_length, d_model]

    def forward(self, input_ids, segment_ids, attention_mask):
        # Input shapes: [batch_size, seq_length, channels]
        # e.g., [3, 1600, 4]
        input_ids = input_ids.long()  # Removed .to(self.device)
        segment_ids = segment_ids.long()  # Removed .to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.float()  # Removed .to(self.device)

        # Validate inputs
        assert input_ids.max() < self.vocab_size, f"input_ids max {input_ids.max()} >= vocab_size {self.vocab_size}"

        # Get dimensions
        batch_size, seq_length, channels = input_ids.shape
        
        # Compute embeddings
        token_embeds = self.token_embedding(input_ids)    # [3, 1600, 4, 256]
        token_embeds = token_embeds.view(token_embeds.size(0), -1, token_embeds.size(-1))  # [3, 6400, 256]
        torch.cuda.empty_cache()
        # Prepare positional encoding
        pos_embeds = self.pos_encoding[:, :seq_length, :]  # [1, 1600, 256]
        pos_embeds = torch.cat([pos_embeds] * 4, dim=1).to(input_ids.device)  # [1, 6400, 256], dynamic device
        # Sum embeddings with proper broadcasting
        embedding_output = token_embeds + pos_embeds
        del token_embeds, pos_embeds
        torch.cuda.empty_cache()

        embedding_output = self.dropout(embedding_output)

        # Pass through encoder
        sequence_output = self.encoder(embedding_output, attention_mask)

        # Extract CLS token (first position of first channel)
        cls_output = sequence_output[:, 0, :]  # [batch_size, d_model]
        # Reshape for pooler: [batch_size, d_model]
        cls_output = cls_output.reshape(batch_size, -1)  # [3, 256]
        pooled_output = self.pooler_activation(self.pooler(cls_output))
        # If you want to maintain channels dimension:
        pooled_output = pooled_output.reshape(batch_size, 1, -1)  # [3, 1, 256]
        sequence_output = self.final(sequence_output)
        pooled_output = self.final(pooled_output)
        return sequence_output, pooled_output

    def forward_pre_training(self, input_ids, segment_ids, attention_mask):
        sequence_output, pooled_output = self.forward(input_ids, segment_ids, attention_mask)
        mlm_logits, nsp_logits = self.pre_training_heads(sequence_output, pooled_output)
        return mlm_logits, nsp_logits