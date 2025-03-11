import torch
import os
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, folder_path, interval_length=32, mask_length=2, sample_rate=50, cache_file="dataset.pt"):
        self.interval_length = interval_length * sample_rate  # in samples
        self.mask_length = mask_length * sample_rate         # in samples
        self.sample_rate = sample_rate
        self.cache_file = cache_file
        self.index_map = []

        if os.path.exists(self.cache_file):
            print(f"Loading index map from {self.cache_file}")
            self.load_index_map()
        else:
            print("Processing raw files and saving dataset index...")
            self.precompute_and_store(folder_path)

        print(f"Dataset contains {len(self.index_map)} samples.")

    def precompute_and_store(self, folder_path):
        """Precompute valid intervals and store index mapping without loading everything into memory."""
        index_map = []
        track_files = []
        
        for file in sorted(os.listdir(folder_path)):  # Sort for consistency
            if file.endswith(".pt"):
                file_path = os.path.join(folder_path, file)
                tensor = torch.load(file_path).squeeze(0).transpose(0, 1)  # Convert (1, 4, T) -> (T, 4)
                track_files.append(file_path)  # Store file path instead of data

                track_length = tensor.shape[0]
                valid_intervals = max((track_length - self.interval_length) // self.sample_rate + 1, 0)

                for i in range(valid_intervals):
                    start_idx = i * self.sample_rate
                    end_idx = start_idx + self.interval_length
                    if end_idx <= track_length:
                        index_map.append((file_path, start_idx))

        torch.save(index_map, self.cache_file)  # Save index map
        self.index_map = index_map

    def load_index_map(self):
        """Load only the index mapping, not the full dataset."""
        self.index_map = torch.load(self.cache_file)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        try:
            file_path, local_offset = self.index_map[idx]  # Get file path and offset
            track_data = torch.load(file_path).squeeze(0).transpose(0, 1)  # Load only necessary track
            interval = track_data[local_offset:local_offset + self.interval_length].clone()
            interval_float = interval.float()

            # Define mask region (centrally located)
            mask_start = (self.interval_length - self.mask_length) // 2
            mask_end = mask_start + self.mask_length

            masked_interval = interval.clone()

            # Denoising strategy: use shuffle (suitable for categorical data)
            noise_type = "shuffle"
            if noise_type == "gaussian":
                noise = torch.normal(
                    mean=interval_float.mean(), 
                    std=interval_float.std(),
                    size=(self.mask_length, interval.shape[1])
                ).to(interval.device)
            elif noise_type == "uniform":
                noise = (torch.rand(self.mask_length, interval.shape[1]) *
                         (interval_float.max() - interval_float.min()) +
                         interval_float.min()).to(interval.device)
            elif noise_type == "shuffle":
                indices = torch.randperm(interval.shape[0])[:self.mask_length]
                noise = interval[indices]
            elif noise_type == "mean":
                mean_value = interval_float.mean(dim=0, keepdim=True)
                noise = mean_value.repeat(self.mask_length, 1).to(interval.device)

            if interval.dtype in [torch.int, torch.long]:
                noise = noise.round().clamp(interval.min(), interval.max()).to(interval.dtype)

            # Replace the center region with noise.
            masked_interval[mask_start:mask_end] = noise

        except IndexError as e:
            print(f"[ERROR] Index {idx} out of bounds!")
            raise e

        return masked_interval, interval[mask_start:mask_end]