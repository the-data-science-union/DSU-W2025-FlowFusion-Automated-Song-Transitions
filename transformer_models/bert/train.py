import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model.model import BERT_model
from data.data_loader import MusicDataset
from .config import *
import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import random
import sys

# Initialize Weights & Biases if enabled
if WANDB_LOGS:
    wandb.init(project="transformer_music", config={
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
dataset = MusicDataset(DATA_PATH, INTERVAL_LENGTH, MASK_LENGTH, SAMPLE_RATE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Initialization
model = BERT_model(
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
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
    predicted_classes = torch.argmax(predicted_masks, dim=-1).cpu().numpy()
    original_masks = original_masks.cpu().numpy()

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

    sample_image_path = f"runs/transformer_outputs/mask_comparison_epoch_{epoch+1}_batch_{batch_idx}.png"
    os.makedirs(os.path.dirname(sample_image_path), exist_ok=True)
    plt.savefig(sample_image_path)
    plt.close()

    if WANDB_LOGS:
        wandb.log({"mask_comparison": wandb.Image(sample_image_path), "epoch": epoch + 1, "batch_idx": batch_idx})

# Gradient and parameter logging
def log_gradients_and_params(model, epoch, batch_idx):
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

    avg_grad_norm = np.mean(list(grad_norms.values())) if grad_norms else 0.0
    max_grad_norm = np.max(list(grad_norms.values())) if grad_norms else 0.0
    avg_param_norm = np.mean(list(param_norms.values())) if param_norms else 0.0

    wandb.log({
        "avg_grad_norm": avg_grad_norm,
        "max_grad_norm": max_grad_norm,
        "avg_param_norm": avg_param_norm,
        "epoch": epoch + 1,
        "batch_idx": batch_idx
    }, commit=False)

# Training Loop
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
                
                optimizer.zero_grad()
                input_ids = masked_intervals.long()
                segment_ids = masked_intervals.long()  # Assuming segment_ids same as input_ids

                # Forward pass
                mlm_logits, nsp_logits = model(input_ids, segment_ids, None)  # Assuming BERT_model returns MLM and NSP logits
                mlm_logits = mlm_logits.reshape(BATCH_SIZE, MAX_SEQ_LENGTH, AUDIO_CHANNELS, VOCAB_SIZE)

                # Extract mask region
                mask_start = (mlm_logits.shape[1] - original_masks.shape[1]) // 2
                mask_end = mask_start + original_masks.shape[1]
                predicted_masks = mlm_logits[:, mask_start:mask_end, :]
                predicted_masks = predicted_masks.permute(0, 3, 1, 2)

                # Compute loss (MLM only for simplicity; adjust if NSP is needed)
                loss = criterion(predicted_masks, original_masks)
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Gradient norm
                batch_grad_norm = sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)
                total_grad_norm += batch_grad_norm

                # Accumulate predictions
                all_predicted_masks.append(predicted_masks.detach().cpu())
                all_original_masks.append(original_masks.detach().cpu())

                pbar.set_postfix(loss=loss.item(), grad_norm=batch_grad_norm)

                # Log per-batch metrics to W&B if enabled
                if WANDB_LOGS:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch_grad_norm": batch_grad_norm,
                        "epoch": epoch + 1,
                        "batch_idx": batch_idx
                    }, commit=False)
                    log_gradients_and_params(model, epoch, batch_idx)

        # Epoch metrics
        avg_loss = total_loss / len(dataloader)
        avg_grad_norm_epoch = total_grad_norm / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm_epoch:.4f}")

        # Scheduler step
        scheduler.step(avg_loss)

        # Log per-epoch metrics to W&B if enabled
        if WANDB_LOGS:
            wandb.log({
                "train_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "avg_grad_norm_epoch": avg_grad_norm_epoch,
                "epoch": epoch + 1
            }, commit=True)

        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"runs/transformer_runs/bert_epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            if WANDB_LOGS:
                wandb.save(checkpoint_path)

        # Audio and visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Concatenate accumulated masks
            all_predicted_masks = torch.cat(all_predicted_masks, dim=0)
            all_original_masks = torch.cat(all_original_masks, dim=0)

            # Random sample for visualization and audio
            num_samples = all_predicted_masks.shape[0]
            random_batch_idx = random.randint(0, num_samples // BATCH_SIZE - 1)
            batch_start = random_batch_idx * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE

            random_predicted_masks = all_predicted_masks[batch_start:batch_end]
            random_original_masks = all_original_masks[batch_start:batch_end]

            # Save visualization
            save_sample_masks(random_predicted_masks, random_original_masks, epoch, random_batch_idx)

            # Generate and log audio
            pred_audio_path = f"runs/transformer_outputs/predicted_epoch_{epoch+1}.wav"
            orig_audio_path = f"runs/transformer_outputs/original_epoch_{epoch+1}.wav"
            os.makedirs(os.path.dirname(pred_audio_path), exist_ok=True)

            pred_indices = torch.argmax(random_predicted_masks, dim=-1)
            pred_audio_file = convert_to_wav(pred_indices, pred_audio_path)
            orig_audio_file = convert_to_wav(random_original_masks, orig_audio_path)

            if WANDB_LOGS:
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