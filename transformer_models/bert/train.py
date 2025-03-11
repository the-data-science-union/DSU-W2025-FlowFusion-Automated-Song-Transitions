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
from data.data_loader_test_2 import test_dataloader

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

test_dataloader(dataset, dataloader)

# Set device and limit to maximum 2 GPUs (using CUDA:2 and CUDA:3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Default to cuda:2
available_gpus = torch.cuda.device_count()
if available_gpus > 3:  # Check if we have at least 4 GPUs (0,1,2,3)
    num_gpus = 2  # Use exactly 2 GPUs (cuda:2 and cuda:3)
    device_ids = [0, 1]  # Specify GPU device IDs 2 and 3
    print(f"Using 2 GPUs: cuda:2 and cuda:3")
else:
    num_gpus = 1 if available_gpus >= 3 else 0  # Use cuda:2 if available, else CPU
    device_ids = [2] if available_gpus >= 3 else None
    print(f"Using single device: {device}")

# Load Pre-trained Model
model_path = "/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/runs/transformer_runs/bert_epoch_135.pt"
model = BERT_model(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=DROPOUT,
).to(device)

# Load the saved state dictionary
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
print(f"Loaded pre-trained model from {model_path} (epoch 135)")

# Wrap the model with DataParallel if multiple GPUs are available (using cuda:2 and cuda:3)
if num_gpus > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

# Rest of the code remains the same...
# Loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Audio conversion function
def convert_to_wav(denorm_data, output_file, device=device):
    model = EncodecModel.encodec_model_48khz().to(device)
    model.set_target_bandwidth(6.0)
    
    if denorm_data.dim() == 2:  # [T, N]
        denorm_data = denorm_data.transpose(0, 1).unsqueeze(0)  # [1, N, T]
    elif denorm_data.dim() == 3:  # [B, T, N]
        denorm_data = denorm_data.transpose(1, 2)  # [B, N, T]
    
    denorm_data = denorm_data.long().to(device)
    encoded_frame = (denorm_data, None)
    with torch.no_grad():
        decoded_audio = model.decode([encoded_frame])  # [B, C, T]
    
    # Save each sample
    for i in range(decoded_audio.shape[0]):
        sample_audio = decoded_audio[i]  # [C, T]
        sample_file = f"{output_file[:-4]}_{i}.wav" if decoded_audio.shape[0] > 1 else output_file
        torchaudio.save(sample_file, sample_audio.cpu(), sample_rate=model.sample_rate)
    return output_file  # Base file name

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
def train(start_epoch=110):
    model.train()

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        total_grad_norm = 0.0
        all_predicted_masks = []
        all_original_masks = []

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for batch_idx, (masked_intervals, original_masks) in enumerate(pbar):
                masked_intervals, original_masks = masked_intervals.to(device), original_masks.to(device)
                
                optimizer.zero_grad()
                input_ids = masked_intervals.long()
                segment_ids = masked_intervals.long()  # Assuming segment_ids same as input_ids

                # Forward pass
                mlm_logits, nsp_logits = model(input_ids, segment_ids, None)
                mlm_logits = mlm_logits.reshape(BATCH_SIZE, MAX_SEQ_LENGTH, AUDIO_CHANNELS, VOCAB_SIZE)
                
                # Extract mask region
                mask_start = (mlm_logits.shape[1] - original_masks.shape[1]) // 2
                mask_end = mask_start + original_masks.shape[1]
                predicted_masks = mlm_logits[:, mask_start:mask_end, :]
                predicted_masks = predicted_masks.permute(0, 3, 1, 2)
                
                # Compute loss
                loss = criterion(predicted_masks, original_masks)
                total_loss += loss.item()
                predictions = torch.argmax(predicted_masks, dim=1)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Gradient norm
                batch_grad_norm = sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)
                total_grad_norm += batch_grad_norm

                # Accumulate predictions
                all_predicted_masks.append(predictions.detach().cpu())
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
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, checkpoint_path)
            if WANDB_LOGS:
                wandb.save(checkpoint_path)

        # Audio and visualization every 5 epochs
        if (epoch) % 5 == 0:
            all_predicted_masks = torch.cat(all_predicted_masks, dim=0)
            all_original_masks = torch.cat(all_original_masks, dim=0)

            num_samples = all_predicted_masks.shape[0]
            random_batch_idx = random.randint(0, num_samples // BATCH_SIZE - 1)
            batch_start = random_batch_idx * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE

            random_predicted_masks = all_predicted_masks[batch_start:batch_end]
            random_original_masks = all_original_masks[batch_start:batch_end]
            
            # Base paths for audio files
            pred_audio_base = f"runs/transformer_outputs/predicted_epoch_{epoch+1}"
            orig_audio_base = f"runs/transformer_outputs/original_epoch_{epoch+1}"
            os.makedirs(os.path.dirname(pred_audio_base), exist_ok=True)
            
            # Generate and save audio files
            pred_audio_file = convert_to_wav(random_predicted_masks, f"{pred_audio_base}.wav", device=device)
            orig_audio_file = convert_to_wav(random_original_masks, f"{orig_audio_base}.wav", device=device)

            if WANDB_LOGS:
                # Collect all generated audio files
                pred_audio_files = []
                orig_audio_files = []
                
                # Check for multiple files (when batch size > 1)
                if random_predicted_masks.shape[0] > 1:
                    for i in range(random_predicted_masks.shape[0]):
                        pred_path = f"{pred_audio_base}_{i}.wav"
                        orig_path = f"{orig_audio_base}_{i}.wav"
                        if os.path.exists(pred_path):
                            pred_audio_files.append(wandb.Audio(
                                pred_path, 
                                caption=f"Predicted Audio Epoch {epoch+1} Sample {i}",
                                sample_rate=48000
                            ))
                        if os.path.exists(orig_path):
                            orig_audio_files.append(wandb.Audio(
                                orig_path,
                                caption=f"Original Audio Epoch {epoch+1} Sample {i}",
                                sample_rate=48000
                            ))
                else:
                    if os.path.exists(pred_audio_file):
                        pred_audio_files.append(wandb.Audio(
                            pred_audio_file,
                            caption=f"Predicted Audio Epoch {epoch+1}",
                            sample_rate=48000
                        ))
                    if os.path.exists(orig_audio_file):
                        orig_audio_files.append(wandb.Audio(
                            orig_audio_file,
                            caption=f"Original Audio Epoch {epoch+1}",
                            sample_rate=48000
                        ))

                # Log all audio files to W&B
                wandb_log_dict = {"epoch": epoch + 1}
                if pred_audio_files:
                    wandb_log_dict["predicted_audio"] = pred_audio_files
                if orig_audio_files:
                    wandb_log_dict["original_audio"] = orig_audio_files
                
                if pred_audio_files or orig_audio_files:
                    wandb.log(wandb_log_dict)

if __name__ == "__main__":
    try:
        train(start_epoch=135)
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