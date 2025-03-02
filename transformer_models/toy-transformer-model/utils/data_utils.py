import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import nltk
from nltk.corpus import brown
import random
import config
import os

nltk.data.path.append(os.path.join(config.DATA_DIR, 'nltk_data'))
nltk.download('brown', download_dir=os.path.join(config.DATA_DIR, 'nltk_data'), quiet=True)

class NLTKDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = config.DEVICE # added this

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = ' '.join(self.sentences[idx])
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0).to(self.device) for key, val in encoding.items()} #this sends it to GPU
        return item

def create_mlm_nsp_inputs(examples, tokenizer, mlm_probability=0.15):
    tokenized_inputs = tokenizer([' '.join(example) for example in examples], padding=True, truncation=True, return_tensors="pt")
    inputs = tokenized_inputs

    # Move inputs to CUDA before further processing
    inputs["input_ids"] = inputs["input_ids"].to(config.DEVICE)
    inputs["attention_mask"] = inputs["attention_mask"].to(config.DEVICE)
    inputs["token_type_ids"] = inputs["token_type_ids"].to(config.DEVICE)
    inputs["labels"] = inputs["input_ids"].clone()

    # Mask tokens for MLM
    probability_matrix = torch.full(inputs["input_ids"].shape, mlm_probability, device=config.DEVICE)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs["input_ids"].tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=config.DEVICE)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs["labels"][~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(inputs["input_ids"].shape, 0.8, device=config.DEVICE)).bool() & masked_indices
    inputs["input_ids"][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(inputs["input_ids"].shape, 0.5, device=config.DEVICE)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(0, len(tokenizer), inputs["input_ids"].shape, dtype=torch.long, device=config.DEVICE)
    inputs["input_ids"][indices_random] = random_words[indices_random]

    # Create next sentence prediction labels (assuming 50% are next sentences)
    inputs["next_sentence_label"] = torch.randint(0, 2, (inputs["input_ids"].shape[0],), device=config.DEVICE)  # Ensure correct device

    return inputs

def get_nltk_dataloader(tokenizer, max_length, batch_size):
    sentences = brown.sents()
    dataset = NLTKDataset(sentences, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
