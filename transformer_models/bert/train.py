import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model.model import BERT_model
from data.data_loader import MusicDataset
import os
from .config import *

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (masked_interval, target) in enumerate(train_loader):
        masked_interval = masked_interval.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()

        input_ids = masked_interval.long()  # This assumes masked_interval is treated as token ids.
        segment_ids = masked_interval.long()  # Using segment_ids to carry the same data in this case.
        attention_mask = (input_ids != 0).long()  # Assuming 0 is padding.

        outputs = model(input_ids, segment_ids, attention_mask)

        # Assuming the outputs are the prediction logits (MLM and NSP).
        mlm_logits, nsp_logits = outputs

        print("mlm_logits shape:", mlm_logits.shape)
        print("target shape:", target.shape)
        
        # The loss could be calculated as a sum of the MLM and NSP loss.
        # Assuming you have labels for MLM and NSP tasks.
        mlm_loss = criterion(mlm_logits.view(-1, mlm_logits.size(-1)), target.view(-1))
        nsp_loss = criterion(nsp_logits.view(-1, nsp_logits.size(-1)), target.view(-1))  # Adjust if needed

        loss = mlm_loss + nsp_loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {running_loss / (batch_idx + 1)}")

    return running_loss / len(train_loader)

def main():
    device = torch.device(DEVICE)

    model = BERT_model(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = MusicDataset(data_path, interval_length=32, mask_length=2, sample_rate=50)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss after Epoch {epoch + 1}: {train_loss}")

if __name__ == "__main__":
    main()
