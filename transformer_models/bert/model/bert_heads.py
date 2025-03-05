import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLanguageModelHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(MaskedLanguageModelHead, self).__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, sequence_output):
        x = self.dense(sequence_output)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class NextSentencePredictionHead(nn.Module):
    def __init__(self, d_model):
        super(NextSentencePredictionHead, self).__init__()
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, pooled_output):
        return self.classifier(pooled_output)

class BERTPreTrainingHeads(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(BERTPreTrainingHeads, self).__init__()
        self.mlm_head = MaskedLanguageModelHead(d_model, vocab_size)
        self.nsp_head = NextSentencePredictionHead(d_model)

    def forward(self, sequence_output, pooled_output):
        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)
        return mlm_logits, nsp_logits
