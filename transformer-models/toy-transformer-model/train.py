import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BERT
from transformers import BertTokenizer

class BERTTrainer:
    def __init__(self, model, vocab_size, max_seq_length, learning_rate=1e-4, warmup_steps=10000):
        self.model = model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss_fn = nn.CrossEntropyLoss()
        
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.warmup_steps - current_step) / float(max(1, self.warmup_steps - self.total_steps)))
        
    def prepare_inputs(self, batch):
        input_ids, segment_ids, attention_mask, masked_lm_labels, next_sentence_labels = batch
        return input_ids.to(self.device), segment_ids.to(self.device), attention_mask.to(self.device), \
               masked_lm_labels.to(self.device), next_sentence_labels.to(self.device)
               
    def train_step(self, batch):
        self.model.train()
        input_ids, segment_ids, attention_mask, masked_lm_labels, next_sentence_labels = self.prepare_inputs(batch)
        
        self.optimizer.zero_grad()
        
        mlm_logits, nsp_logits = self.model.forward_pre_training(input_ids, segment_ids, attention_mask)
        
        mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        nsp_loss = self.nsp_loss_fn(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))
        
        total_loss = mlm_loss + nsp_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss.item(), mlm_loss.item(), nsp_loss.item()
        
    def train(self, train_dataloader, num_epochs):
        self.model.to(self.device)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_mlm_loss = 0
            epoch_nsp_loss = 0
            
            for batch in train_dataloader:
                batch_loss, batch_mlm_loss, batch_nsp_loss = self.train_step(batch)
                epoch_loss += batch_loss
                epoch_mlm_loss += batch_mlm_loss
                epoch_nsp_loss += batch_nsp_loss
                
            avg_loss = epoch_loss / len(train_dataloader)
            avg_mlm_loss = epoch_mlm_loss / len(train_dataloader)
            avg_nsp_loss = epoch_nsp_loss / len(train_dataloader)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average MLM Loss: {avg_mlm_loss:.4f}")
            print(f"Average NSP Loss: {avg_nsp_loss:.4f}")
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

# Example usage
vocab_size = 30522  # BERT's default vocab size
d_model = 768
num_layers = 12
num_heads = 12
d_ff = 3072
max_seq_length = 512
dropout = 0.1

bert_model = BERT(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
trainer = BERTTrainer(bert_model, vocab_size, max_seq_length)

# Assuming you have a DataLoader for your training data
# train_dataloader = DataLoader(...)

# trainer.train(train_dataloader, num_epochs=3)
# trainer.save_model("bert_model.pth")
