from transformers import BertTokenizer
from utils.data_utils import get_nltk_dataloader, create_mlm_nsp_inputs
from model.model import BERT
from train import BERTTrainer
import config
import torch
import os
from torch.utils.data import random_split

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Get full dataloader
    full_dataloader = get_nltk_dataloader(tokenizer, config.MAX_LENGTH, config.BATCH_SIZE)

    # Extract the dataset from the dataloader
    full_dataset = full_dataloader.dataset  # <-- Extract dataset before splitting

    # Now perform the train-validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create new dataloaders from the split datasets
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    
    def collate_fn(examples):
        return create_mlm_nsp_inputs(examples, tokenizer)

    train_dataloader.collate_fn = collate_fn
    val_dataloader.collate_fn = collate_fn

    vocab_size = 30522  # BERT's default vocab size
    d_model = 768
    num_layers = 12
    num_heads = 12
    d_ff = 3072
    max_seq_length = 512
    dropout = 0.1

    bert_model = BERT(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
    trainer = BERTTrainer(bert_model, vocab_size, max_seq_length, config.LEARNING_RATE, config.WARMUP_STEPS)

    print(f"Length of train dataloader: {len(train_dataloader)}")
    print(f"Length of validation dataloader: {len(val_dataloader)}")

    trainer.train(train_dataloader, val_dataloader, config.NUM_EPOCHS)

if __name__ == "__main__":
    main()