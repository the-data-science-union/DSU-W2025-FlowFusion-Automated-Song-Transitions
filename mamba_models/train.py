import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import MusicDataset
from mamba_models.bidrectional_mamba import BidirectionalMamba
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

# Define Training Parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 1
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
NUM_LAYERS = 24
NUM_HEADS = 4
D_FF = 2048
MAX_SEQ_LENGTH = 1600  # Adjust based on data

model = BidirectionalMamba(
    vocab_size=VOCAB_SIZE, 
    d_model=D_MODEL, 
    num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS, 
    d_ff=D_FF, 
    max_seq_length=MAX_SEQ_LENGTH
).to(DEVICE)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Function to save a sample of the predicted mask and original mask
def save_sample_masks(predicted_masks, original_masks, epoch):
    # Convert tensors to numpy arrays for saving as images
    predicted_masks = predicted_masks.squeeze().cpu().numpy()
    original_masks = original_masks.squeeze().cpu().numpy()

    # Create a plot to visualize the predicted vs. original mask
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_masks, cmap='viridis', aspect='auto')
    plt.title("Original Mask")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_masks, cmap='viridis', aspect='auto')
    plt.title("Predicted Mask")
    plt.colorbar()

    # Save the figure as an image
    sample_image_path = f"mask_comparison_epoch_{epoch+1}.png"
    plt.savefig(sample_image_path)
    plt.close()

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

                predicted_masks = outputs[:, mask_start:mask_end, :].float()  # Extract masked region

                # Convert to float as the criterion expects float type
                original_masks = original_masks.float()

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

        # Save model checkpoint and sample masks every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"bidirectional_mamba_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            # Save sample mask comparison for the current epoch
            with torch.no_grad():
                for masked_intervals, original_masks in dataloader:
                    masked_intervals, original_masks = masked_intervals.to(DEVICE), original_masks.to(DEVICE)
                    outputs = model(masked_intervals)
                    mask_start = (outputs.shape[1] - original_masks.shape[1]) // 2
                    mask_end = mask_start + original_masks.shape[1]
                    predicted_masks = outputs[:, mask_start:mask_end, :].float()
                    save_sample_masks(predicted_masks, original_masks, epoch)
                    break  # Save only one batch sample per 5 epochs

if __name__ == "__main__":
    train()
