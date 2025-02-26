import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import MusicDataset
from mamba_models.bidrectional_mamba import BidirectionalMamba

# Define Training Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
INTERVAL_LENGTH = 32
MASK_LENGTH = 2
SAMPLE_RATE = 50
FILE_PATH = "/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/data/processed-tokens/"

# Initialize Dataset & DataLoader
dataset = MusicDataset(FILE_PATH, INTERVAL_LENGTH, MASK_LENGTH, SAMPLE_RATE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Initialization
VOCAB_SIZE = 32759  # Update if different
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 8
D_FF = 512
MAX_SEQ_LENGTH = 1600  # Adjust based on data

model = BidirectionalMamba(
    vocab_size=VOCAB_SIZE, 
    d_model=D_MODEL, 
    num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS, 
    d_ff=D_FF, 
    max_seq_length=MAX_SEQ_LENGTH
).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for masked_intervals, original_masks in dataloader:
            masked_intervals, original_masks = masked_intervals.to(DEVICE), original_masks.to(DEVICE)

            # Forward pass
            outputs = model(masked_intervals)  # Shape: (B, W, D)
            mask_start = (outputs.shape[1] - original_masks.shape[1]) // 2
            mask_end = mask_start + original_masks.shape[1]

            predicted_masks = outputs[:, mask_start:mask_end, :]  # Extract masked region

            # Compute loss
            loss = criterion(predicted_masks, original_masks)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"bidirectional_mamba_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
