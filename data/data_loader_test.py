import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data_loader import MusicDataset

def test_music_dataset(file_path, interval_length=32, mask_length=2, sample_rate=25, batch_size=1):
    dataset = MusicDataset(file_path, interval_length, mask_length, sample_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    masked_intervals, original_masks = next(iter(dataloader))

    print(f"Masked intervals shape: {masked_intervals.shape}")
    print(f"Original masks shape: {original_masks.shape}")

    expected_interval_length = interval_length * sample_rate
    expected_mask_length = mask_length * sample_rate
    assert masked_intervals.shape == (batch_size, expected_interval_length, 4), "Incorrect masked interval shape"
    assert original_masks.shape == (batch_size, expected_mask_length, 4), "Incorrect original mask shape"

    # data viz
    sample_idx = 0
    visualize_sample(masked_intervals[sample_idx], original_masks[sample_idx], sample_rate)

    mask_start = (expected_interval_length - expected_mask_length) // 2
    mask_end = mask_start + expected_mask_length

    print("All tests passed successfully!")

def visualize_sample(masked_interval, original_mask, sample_rate):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(2, 1, 1)
    plt.imshow(masked_interval.numpy().reshape(1, -1), aspect='auto', cmap='viridis')
    plt.title("Masked Interval")
    plt.xlabel("Time")
    plt.ylabel("Channels")
    mask_start = (len(masked_interval) - len(original_mask)) // 2
    mask_end = mask_start + len(original_mask)
    plt.axvline(x=mask_start, color='r', linestyle='--')
    plt.axvline(x=mask_end, color='r', linestyle='--')
    plt.subplot(2, 1, 2)
    plt.imshow(original_mask.numpy().reshape(1, -1), aspect='auto', cmap='viridis')
    plt.title("Original Mask")
    plt.xlabel("Time")
    plt.ylabel("Channels")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/data/processed-tokens/"
    test_music_dataset(file_path)