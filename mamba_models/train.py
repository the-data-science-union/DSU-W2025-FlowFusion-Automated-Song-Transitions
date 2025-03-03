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
import wandb
import sys
import random

load_dotenv()

# Initialize Weights & Biases
wandb.init(project="bidirectional_mamba_music", config={
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "interval_length": 32,
    "mask_length": 2,
    "sample_rate": 50,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 16,
    "d_ff": 1024,
    "max_seq_length": 1600,
    "dropout": 0.3
})

# Define Training Parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = wandb.config.epochs
BATCH_SIZE = wandb.config.batch_size
LEARNING_RATE = wandb.config.learning_rate
INTERVAL_LENGTH = wandb.config.interval_length
MASK_LENGTH = wandb.config.mask_length
SAMPLE_RATE = wandb.config.sample_rate
FILE_PATH = "/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/data/processed-tokens/"

# Initialize Dataset & DataLoader
dataset = MusicDataset(FILE_PATH, INTERVAL_LENGTH, MASK_LENGTH, SAMPLE_RATE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Initialization
VOCAB_SIZE = 10000
D_MODEL = wandb.config.d_model
NUM_LAYERS = wandb.config.num_layers
NUM_HEADS = wandb.config.num_heads
D_FF = wandb.config.d_ff
MAX_SEQ_LENGTH = wandb.config.max_seq_length
DROPOUT = wandb.config.dropout

model = BidirectionalMamba(
    vocab_size=VOCAB_SIZE, 
    d_model=D_MODEL, 
    num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS, 
    d_ff=D_FF, 
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=DROPOUT
).to(DEVICE)

# Loss function, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Function to save and log sample masks
def save_sample_masks(predicted_masks, original_masks, epoch, batch_idx):
    predicted_masks = predicted_masks.squeeze().cpu().numpy()
    original_masks = original_masks.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_masks, cmap='viridis', aspect='auto')
    plt.title("Original Mask")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_masks, cmap='viridis', aspect='auto')
    plt.title("Predicted Mask")
    plt.colorbar()

    sample_image_path = f"mask_comparison_epoch_{epoch+1}_batch_{batch_idx}.png"
    plt.savefig(sample_image_path)
    plt.close()

    wandb.log({"mask_comparison": wandb.Image(sample_image_path), "epoch": epoch + 1})

# Function to log gradients and parameter statistics
def log_gradients_and_params(model, epoch):
    grad_norms = {}
    param_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()
            grad_norms[f"grad_norm/{name}"] = grad_norm
            param_norms[f"param_norm/{name}"] = param.norm(2).item()
    
    wandb.log(grad_norms, commit=False)
    wandb.log(param_norms, commit=False)

    avg_grad_norm = np.mean(list(grad_norms.values()))
    max_grad_norm = np.max(list(grad_norms.values()))
    avg_param_norm = np.mean(list(param_norms.values()))
    
    wandb.log({
        "avg_grad_norm": avg_grad_norm,
        "max_grad_norm": max_grad_norm,
        "avg_param_norm": avg_param_norm,
        "epoch": epoch + 1
    }, commit=False)

# Training Loop with Enhanced Logging and Accumulation
def train():
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_grad_norm = 0.0
        all_predicted_masks = []
        all_original_masks = []

        all_predicted_masks = []
        all_original_masks = []
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for batch_idx, (masked_intervals, original_masks) in enumerate(pbar):
                masked_intervals, original_masks = masked_intervals.to(DEVICE), original_masks.to(DEVICE)

                # Forward pass
                outputs = model(masked_intervals)
                mask_start = (outputs.shape[1] - original_masks.shape[1]) // 2
                mask_end = mask_start + original_masks.shape[1]
                predicted_masks = outputs[:, mask_start:mask_end, :].float()
                original_masks = original_masks.float()

                # Accumulate predictions and ground truth
                all_predicted_masks.append(predicted_masks.detach().cpu())
                all_original_masks.append(original_masks.detach().cpu())

                # Compute loss
                loss = criterion(predicted_masks, original_masks)
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()

                # Compute gradient norm for this batch
                batch_grad_norm = sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)
                total_grad_norm += batch_grad_norm

                pbar.set_postfix(loss=loss.item(), grad_norm=batch_grad_norm)

        # Average metrics over the epoch
        avg_loss = total_loss / len(dataloader)
        avg_grad_norm_epoch = total_grad_norm / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm_epoch:.4f}")

        # Step the scheduler based on average loss
        scheduler.step(avg_loss)

        # Log per-epoch metrics to W&B (without mask stats yet)
        wandb.log({
            "train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "avg_grad_norm_epoch": avg_grad_norm_epoch,
            "epoch": epoch + 1,
        }, commit=True)

        # Log gradients and parameter statistics
        log_gradients_and_params(model, epoch)

        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"bidirectional_mamba_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)

        # After training, concatenate all accumulated masks
        all_predicted_masks = torch.cat(all_predicted_masks, dim=0)
        all_original_masks = torch.cat(all_original_masks, dim=0)

        # Calculate statistics for all predicted and original masks
        output_mean = all_predicted_masks.mean().item()
        output_std = all_predicted_masks.std().item()
        expected_mean = all_original_masks.mean().item()
        expected_std = all_original_masks.std().item()

        # Log the final aggregated stats to W&B
        wandb.log({
            "output_mean": output_mean,
            "output_std": output_std,
            "expected_mean": expected_mean,
            "expected_std": expected_std,
        })

        if (epoch % 5 == 0):

            # Randomly select a batch from the accumulated masks
            num_samples = all_predicted_masks.shape[0]
            random_batch_idx = random.randint(0, num_samples // BATCH_SIZE - 1)
            batch_start = random_batch_idx * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE

            random_predicted_masks = all_predicted_masks[batch_start:batch_end]
            random_original_masks = all_original_masks[batch_start:batch_end]

            # Save and visualize the random batch
            save_sample_masks(random_predicted_masks, random_original_masks, epoch=EPOCHS-1, batch_idx=random_batch_idx)

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted by user. Closing W&B session...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        wandb.finish()
        sys.exit(0)