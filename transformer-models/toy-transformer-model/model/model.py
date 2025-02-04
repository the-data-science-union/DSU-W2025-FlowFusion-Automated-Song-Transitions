import torch
import torch.nn as nn
from encoder import Encoder
from embeddings import BERTEmbedding
from bert_heads import BERTPreTrainingHeads

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(BERT, self).__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_length, dropout)
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()
        self.pre_training_heads = BERTPreTrainingHeads(d_model, vocab_size)
        
    def forward(self, input_ids, segment_ids, attention_mask):
        embedding_output = self.embedding(input_ids, segment_ids)
        sequence_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        return sequence_output, pooled_output

    def get_input_embeddings(self):
        return self.embedding.token_embed

    def set_input_embeddings(self, value):
        self.embedding.token_embed = value

    def forward_pre_training(self, input_ids, segment_ids, attention_mask):
        sequence_output, pooled_output = self.forward(input_ids, segment_ids, attention_mask)
        mlm_logits, nsp_logits = self.pre_training_heads(sequence_output, pooled_output)
        return mlm_logits, nsp_logits
