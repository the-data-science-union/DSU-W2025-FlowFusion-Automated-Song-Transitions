import math
import torch
import torch.nn as nn
from .encoder import Encoder
from .embeddings import BERTEmbedding
from .bert_heads import BERTPreTrainingHeads
import config

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(BERT, self).__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_length, dropout)
        self.pos_encoding = self.positional_encoding(max_seq_length, d_model)
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()
        self.pre_training_heads = BERTPreTrainingHeads(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.to(config.DEVICE)

    def forward(self, input_ids, segment_ids, attention_mask):
        # Ensure input_ids and segment_ids are Long tensors
        input_ids = input_ids.long()
        segment_ids = segment_ids.long()
        attention_mask = attention_mask.long()
        
        # Get word embeddings
        embedding_output = self.embedding(input_ids, segment_ids)
        
        # Get positional embeddings
        seq_length = input_ids.size(1)
        pos_encodings = self.pos_encoding[:, :seq_length, :].to(config.DEVICE)
        embedding_output = embedding_output + pos_encodings
        
        # Pass the combined embeddings to the encoder
        sequence_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        return sequence_output, pooled_output

    def positional_encoding(self, max_seq_length, d_model):
        pos_encoding = torch.zeros(1, max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def get_input_embeddings(self):
        return self.embedding.token_embed

    def set_input_embeddings(self, value):
        self.embedding.token_embed = value

    def forward_pre_training(self, input_ids, segment_ids, attention_mask):
        sequence_output, pooled_output = self.forward(input_ids, segment_ids, attention_mask)
        mlm_logits, nsp_logits = self.pre_training_heads(sequence_output, pooled_output)
        return mlm_logits, nsp_logits
