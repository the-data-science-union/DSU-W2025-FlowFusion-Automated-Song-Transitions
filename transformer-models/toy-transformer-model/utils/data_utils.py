import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import nltk
from nltk.corpus import brown
import random

nltk.download('brown')

class NLTKDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = ' '.join(self.sentences[idx])
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}

def create_mlm_nsp_inputs(batch, tokenizer, mlm_probability=0.15):
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    
    # Mask tokens for MLM
    probability_matrix = torch.full(inputs["input_ids"].shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs["input_ids"].tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs["labels"][~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(inputs["input_ids"].shape, 0.8)).bool() & masked_indices
    inputs["input_ids"][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(inputs["input_ids"].shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), inputs["input_ids"].shape, dtype=torch.long)
    inputs["input_ids"][indices_random] = random_words[indices_random]

    # Create next sentence prediction labels (assuming 50% are next sentences)
    inputs["next_sentence_label"] = torch.tensor([random.randint(0, 1) for _ in range(inputs["input_ids"].shape[0])])

    return inputs

def get_nltk_dataloader(tokenizer, max_length, batch_size):
    sentences = brown.sents()
    dataset = NLTKDataset(sentences, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
