import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import MusicDataset
from mamba_models.bidrectional_mamba import BidirectionalMamba
from tqdm import tqdm

# Define Training Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
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
NUM_HEADS = 4
D_FF = 1024
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
        total_loss = 0.0  # Total loss for the current epoch
        
        # Use tqdm for the epoch progress bar
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for masked_intervals, original_masks in pbar:
                # Move batches to the device (GPU or CPU)
                masked_intervals, original_masks = masked_intervals.to(DEVICE), original_masks.to(DEVICE)

                # Forward pass
                outputs = model(masked_intervals)  # Shape: (B, W, D)
                mask_start = (outputs.shape[1] - original_masks.shape[1]) // 2
                mask_end = mask_start + original_masks.shape[1]

                predicted_masks = outputs[:, mask_start:mask_end, :]  # Extract masked region

                # Squeeze the last dimension for processing
                predicted_masks = predicted_masks.squeeze(-1)
                
                # Linear layer to map the output to the correct shape
                linear = nn.Linear(1024, 4).to(DEVICE)

                # Convert to float as the criterion expects float type
                original_masks = original_masks.float()
                predicted_masks = linear(predicted_masks).float()

                # Compute loss (Mean Squared Error Loss)
                loss = criterion(predicted_masks, original_masks)
                total_loss += loss.item()  # Accumulate loss for the epoch

                # Backward pass and optimizer step
                optimizer.zero_grad()  # Reset gradients from the previous step
                loss.backward()        # Compute gradients for the current mini-batch
                optimizer.step()       # Update model parameters based on gradients

                # Update the progress bar with the current loss
                pbar.set_postfix(loss=loss.item())

        # Compute the average loss over all batches for the current epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"bidirectional_mamba_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
