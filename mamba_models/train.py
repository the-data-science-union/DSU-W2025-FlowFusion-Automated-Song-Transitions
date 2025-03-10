import os
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import MusicDataset
from mamba_models.bidrectional_mamba import BidirectionalMamba
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from .config import *

load_dotenv()

# Initialize Weights & Biases if enabled
if WANDB_LOGS:
    wandb.init(project="bidirectional_mamba_music", config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "interval_length": INTERVAL_LENGTH,
        "mask_length": MASK_LENGTH,
        "sample_rate": SAMPLE_RATE,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": D_FF,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dropout": DROPOUT
    })

# Initialize Dataset & DataLoader
dataset = MusicDataset(FILE_PATH, INTERVAL_LENGTH, MASK_LENGTH, SAMPLE_RATE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BidirectionalMamba(
    vocab_size=VOCAB_SIZE, 
    d_model=D_MODEL, 
    num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS, 
    d_ff=D_FF, 
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=DROPOUT,
    device=DEVICE
).to(DEVICE)

# Loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Audio conversion function
def convert_to_wav(denorm_data, output_file, device=DEVICE):
    model = EncodecModel.encodec_model_48khz().to(device)
    model.set_target_bandwidth(6.0)
    
    if denorm_data.dim() == 2:  # [T, N]
        denorm_data = denorm_data.transpose(0, 1).unsqueeze(0)  # [1, N, T]
    elif denorm_data.dim() == 3:  # [B, T, N]
        denorm_data = denorm_data.transpose(1, 2)  # [B, N, T]
    
    denorm_data = denorm_data.long().to(device)
    encoded_frame = (denorm_data, None)
    with torch.no_grad():
        decoded_audio = model.decode([encoded_frame])
    
    torchaudio.save(output_file, decoded_audio.squeeze(0).cpu(), sample_rate=model.sample_rate)
    return output_file

# Visualization function
def save_sample_masks(predicted_masks, original_masks, epoch, batch_idx):
    predicted_classes = torch.argmax(predicted_masks, dim=-1).cpu().numpy()  # [batch, mask_len]
    original_masks = original_masks.cpu().numpy()  # [batch, mask_len]

    if predicted_classes.ndim == 1:
        predicted_classes = predicted_classes.reshape(1, -1)
    if original_masks.ndim == 1:
        original_masks = original_masks.reshape(1, -1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_masks, cmap='viridis', aspect='auto')
    plt.title("Original Mask Classes")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_classes, cmap='viridis', aspect='auto')
    plt.title("Predicted Mask Classes")
    plt.colorbar()

    sample_image_path = f"runs/mamba_outputs/data_images/mask_comparison_epoch_{epoch+1}_batch_{batch_idx}.png"
    os.makedirs(os.path.dirname(sample_image_path), exist_ok=True)
    plt.savefig(sample_image_path)
    plt.close()

    if WANDB_LOGS:
        wandb.log({"mask_comparison": wandb.Image(sample_image_path), "epoch": epoch + 1})

# Function to log gradients and parameter statistics
def log_gradients_and_params(model, epoch):
    if not WANDB_LOGS:
        return
    
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

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for batch_idx, (masked_intervals, original_masks) in enumerate(pbar):
                masked_intervals, original_masks = masked_intervals.to(DEVICE), original_masks.to(DEVICE)

                # Forward pass
                outputs = model(masked_intervals)  # [batch, seq_len, vocab_size]
                mask_start = (outputs.shape[1] - original_masks.shape[1]) // 2
                mask_end = mask_start + original_masks.shape[1]
                predicted_masks = outputs[:, mask_start:mask_end, :]  # [batch, mask_len, vocab_size]

                flatten = nn.Flatten(start_dim=-2)
                original_masks_flat = flatten(original_masks)
                predicted_masks_flat = predicted_masks.view(predicted_masks.size(0), -1, predicted_masks.size(-1))
                predicted_masks_flat = predicted_masks_flat.transpose(1, 2)

                # Accumulate predictions and ground truth
                all_predicted_masks.append(predicted_masks.detach().cpu())
                all_original_masks.append(original_masks.detach().cpu())

                # Compute loss
                loss = criterion(predicted_masks_flat, original_masks_flat)
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

        # Log per-epoch metrics to W&B if enabled
        if WANDB_LOGS:
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
            checkpoint_path = f"runs/mamba_runs/bidirectional_mamba_epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            if WANDB_LOGS:
                wandb.save(checkpoint_path)

        # Audio and visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Concatenate accumulated masks
            all_predicted_masks = torch.cat(all_predicted_masks, dim=0)  # [total_samples, mask_len, vocab_size]
            all_original_masks = torch.cat(all_original_masks, dim=0)    # [total_samples, mask_len, 4]

            # Randomly select a batch from the accumulated masks
            num_samples = all_predicted_masks.shape[0]
            random_batch_idx = random.randint(0, num_samples // BATCH_SIZE - 1)
            batch_start = random_batch_idx * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE

            random_predicted_masks = all_predicted_masks[batch_start:batch_end]  # [batch, mask_len, vocab_size]
            random_original_masks = all_original_masks[batch_start:batch_end]    # [batch, mask_len, 4]

            # For visualization, take the first channel of original_masks
            save_sample_masks(random_predicted_masks, random_original_masks[..., 0], epoch, random_batch_idx)

            # Generate and log audio if W&B is enabled
            if WANDB_LOGS:
                pred_audio_path = f"runs/mamba_outputs/predicted_epoch_{epoch+1}.wav"
                orig_audio_path = f"runs/mamba_outputs/original_epoch_{epoch+1}.wav"
                os.makedirs(os.path.dirname(pred_audio_path), exist_ok=True)

                pred_indices = torch.argmax(random_predicted_masks, dim=-1)
                pred_audio_file = convert_to_wav(pred_indices, pred_audio_path)
                orig_audio_file = convert_to_wav(random_original_masks[..., 0], orig_audio_path)  # Using first channel

                wandb.log({
                    "predicted_audio": wandb.Audio(pred_audio_file, caption=f"Predicted Audio Epoch {epoch+1}", sample_rate=48000),
                    "original_audio": wandb.Audio(orig_audio_file, caption=f"Original Audio Epoch {epoch+1}", sample_rate=48000),
                    "epoch": epoch + 1
                })

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if WANDB_LOGS:
            print("Closing W&B session...")
            wandb.finish()
    except Exception as e:
        print(f"An error occurred: {e}")
        if WANDB_LOGS:
            wandb.finish()
    finally:
        if WANDB_LOGS:
            wandb.finish()
        sys.exit(0)