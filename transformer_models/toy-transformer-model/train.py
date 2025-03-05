import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import BERT
from transformers import BertTokenizer
import config
from torch.optim.lr_scheduler import LambdaLR
import os
from tqdm import tqdm


class BERTTrainer:
    def __init__(self, model, vocab_size, max_seq_length, learning_rate=1e-4, warmup_steps=10000):
        self.model = model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-6,
                                     weight_decay=0.01)

        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss_fn = nn.CrossEntropyLoss()
        self.warmup_steps = warmup_steps
        self.model = model.to(config.DEVICE)
        print(f"Model moved to device: {config.DEVICE}")  # Check device

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.warmup_steps - current_step) / float(max(1, self.warmup_steps)))

    def train_step(self, batch):
        try:
            self.model.train()

            input_ids = batch["input_ids"].to(config.DEVICE).long()
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            token_type_ids = batch["token_type_ids"].to(config.DEVICE)
            masked_lm_labels = batch["labels"].to(config.DEVICE)
            next_sentence_labels = batch["next_sentence_label"].to(config.DEVICE)

            mlm_logits, nsp_logits = self.model.forward_pre_training(input_ids, token_type_ids, attention_mask)
            mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, self.vocab_size), masked_lm_labels.view(-1))
            nsp_loss = self.nsp_loss_fn(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))
            total_loss = mlm_loss + nsp_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            return total_loss.item(), mlm_loss.item(), nsp_loss.item()

        except Exception as e:
            print(f"Error in train_step: {e}")
            return 0.0, 0.0, 0.0

    def validate(self, val_dataloader):
        self.model.eval()
        total_loss = 0
        total_mlm_loss = 0
        total_nsp_loss = 0
        num_batches = 0  # Track number of batches

        with torch.no_grad():
            for batch in val_dataloader:
                num_batches += 1  # Count batches
                input_ids = batch["input_ids"].to(config.DEVICE).long()
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                token_type_ids = batch["token_type_ids"].to(config.DEVICE)
                masked_lm_labels = batch["labels"].to(config.DEVICE)
                next_sentence_labels = batch["next_sentence_label"].to(config.DEVICE)

                mlm_logits, nsp_logits = self.model.forward_pre_training(input_ids, token_type_ids, attention_mask)
                mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, self.vocab_size), masked_lm_labels.view(-1))
                nsp_loss = self.nsp_loss_fn(nsp_logits.view(-1, 2), next_sentence_labels.view(-1))
                loss = mlm_loss + nsp_loss

                total_loss += loss.item()
                total_mlm_loss += mlm_loss.item()
                total_nsp_loss += nsp_loss.item()

        if num_batches == 0:  # Prevent division by zero
            return 0.0, 0.0, 0.0

        avg_loss = total_loss / num_batches
        avg_mlm_loss = total_mlm_loss / num_batches
        avg_nsp_loss = total_nsp_loss / num_batches

        return avg_loss, avg_mlm_loss, avg_nsp_loss


    def train(self, train_dataloader, val_dataloader, num_epochs):
        self.model.to(config.DEVICE)

        total_steps = len(train_dataloader) * num_epochs
        self.total_steps = total_steps
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        print(f"Number of epochs: {num_epochs}")
        print(f"Total training steps: {total_steps}")

        for epoch in range(num_epochs):
            epoch_train_loss = 0
            epoch_train_mlm_loss = 0
            epoch_train_nsp_loss = 0

            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

            try:
                for batch in progress_bar:
                    batch_train_loss, batch_train_mlm_loss, batch_train_nsp_loss = self.train_step(batch)
                    epoch_train_loss += batch_train_loss
                    epoch_train_mlm_loss += batch_train_mlm_loss
                    epoch_train_nsp_loss += batch_train_nsp_loss

                    progress_bar.set_postfix({
                        'Train Loss': batch_train_loss,
                        'MLM Train Loss': batch_train_mlm_loss,
                        'NSP Train Loss': batch_train_nsp_loss
                    })

            except Exception as e:
                print(f"Error in epoch {epoch + 1}: {e}, {e.args}")

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            avg_train_mlm_loss = epoch_train_mlm_loss / len(train_dataloader)
            avg_train_nsp_loss = epoch_train_nsp_loss / len(train_dataloader)

            # Validation after each epoch
            val_avg_total_loss, val_avg_mlm_loss, val_avg_nsp_loss = self.validate(val_dataloader)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Average Train Loss: {avg_train_loss:.4f}")
            print(f"Average Train MLM Loss: {avg_train_mlm_loss:.4f}")
            print(f"Average Train NSP Loss: {avg_train_nsp_loss:.4f}")
            print(f"Average Validation Loss: {val_avg_total_loss:.4f}")
            print(f"Average Validation MLM Loss: {val_avg_mlm_loss:.4f}")
            print(f"Average Validation NSP Loss: {val_avg_nsp_loss:.4f}")

    def save_model(self, path):
        path = os.path.join(config.OUTPUT_DIR, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
