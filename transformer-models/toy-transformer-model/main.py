from transformers import BertTokenizer
from utils.data_utils import get_nltk_dataloader, create_mlm_nsp_inputs #Import preprocessing
from model.model import BERT
from train import BERTTrainer
import config
import torch
import os

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataloader = get_nltk_dataloader(tokenizer, config.MAX_LENGTH, config.BATCH_SIZE) # Get the train data
    
    #Data Collator: Applies the mlm and nsp tasks to the batch
    def collate_fn(examples):
        return create_mlm_nsp_inputs(examples, tokenizer)

    train_dataloader.collate_fn = collate_fn  # Use the custom collate function

    vocab_size = 30522  # BERT's default vocab size
    d_model = 768
    num_layers = 12
    num_heads = 12
    d_ff = 3072
    max_seq_length = 512
    dropout = 0.1

    bert_model = BERT(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
    trainer = BERTTrainer(bert_model, vocab_size, max_seq_length, config.LEARNING_RATE, config.WARMUP_STEPS)

    print(f"Length of train dataloader: {len(train_dataloader)}") #Check the data is loaded.

    trainer.train(train_dataloader, config.NUM_EPOCHS)
    trainer.save_model("bert_model.pth")

if __name__ == "__main__":
    main()