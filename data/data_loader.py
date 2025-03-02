import torch
import os
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, folder_path, interval_length=32, mask_length=2, sample_rate=50):
        self.interval_length = interval_length * sample_rate  # Convert to samples
        self.mask_length = mask_length * sample_rate  # Convert to samples
        self.sample_rate = sample_rate

        self.data_files = self.load_all_files(folder_path)  # List of tensors
        self.track_ranges = self.compute_track_ranges()  # Track start indices

        print(f"Loaded {len(self.data_files)} tracks.")
        for i, (start, end) in enumerate(self.track_ranges):
            print(f"Track {i}: {start} to {end} (Length: {end - start})")

    def load_all_files(self, folder_path):
        """Loads all .pt files and keeps them as a list of tensors"""
        data_files = []
        for file in sorted(os.listdir(folder_path)):  # Sort for consistency
            if file.endswith(".pt"):
                file_path = os.path.join(folder_path, file)
                tensor = torch.load(file_path)

                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"File {file} did not load as a tensor. Got {type(tensor)}")

                tensor = tensor.squeeze(0).transpose(0, 1)  # (1, 4, T) → (T, 4)
                print("data loader tensor shape: ", tensor.shape)
                data_files.append(tensor)
                print(f"Loaded {file}, shape: {tensor.shape}")

        if not data_files:
            raise ValueError("No valid .pt files found in the folder!")

        return data_files

    def compute_track_ranges(self):
        """Computes the index ranges for each track to avoid incorrect slicing."""
        track_ranges = []
        current_index = 0

        for tensor in self.data_files:
            track_length = tensor.shape[0]  # T dimension
            track_ranges.append((current_index, current_index + track_length))
            current_index += track_length

        return track_ranges

    def __len__(self):
        """Total valid samples across all tracks"""
        total_samples = 0
        for start, end in self.track_ranges:
            valid_length = (end - start - self.interval_length) // self.sample_rate + 1
            total_samples += max(valid_length, 0)  # Ensure non-negative

        return total_samples

    def get_track_index(self, sample_idx):
        """Finds which track a sample index belongs to."""
        for track_idx, (start, end) in enumerate(self.track_ranges):
            if start <= sample_idx < end:
                return track_idx, sample_idx - start
        raise ValueError(f"Sample index {sample_idx} out of range!")

    import torch

    def __getitem__(self, idx):
        try:
            """Maps dataset index to the correct track & sample location"""
            sample_offset = idx * self.sample_rate

            track_idx, local_idx = self.get_track_index(sample_offset)
            track_data = self.data_files[track_idx]
            end = local_idx + self.interval_length

            # Ensure the interval does not exceed track length
            if end > track_data.shape[0]:  
                local_idx = track_data.shape[0] - self.interval_length
                end = track_data.shape[0]

            interval = track_data[local_idx:end]  # Keep original dtype (likely `torch.long`)
            
            # Convert only for numerical calculations
            interval_float = interval.float()

            # Calculate mask region
            mask_start = (self.interval_length - self.mask_length) // 2
            mask_end = mask_start + self.mask_length

            # Create the masked interval
            masked_interval = interval.clone()

            # Apply denoising strategy (adjusted)
            noise_type = "shuffle"  # Use "shuffle" for categorical data

            if noise_type == "gaussian":
                noise = torch.normal(mean=interval_float.mean(), std=interval_float.std(), 
                                    size=(self.mask_length, interval.shape[1])).to(interval.device)
            elif noise_type == "uniform":
                noise = (torch.rand(self.mask_length, interval.shape[1]) * 
                        (interval_float.max() - interval_float.min()) + interval_float.min()).to(interval.device)
            elif noise_type == "shuffle":
                indices = torch.randperm(interval.shape[0])[:self.mask_length]
                noise = interval[indices]  # Shuffle from sequence
            elif noise_type == "mean":
                mean_value = interval_float.mean(dim=0, keepdim=True)
                noise = mean_value.repeat(self.mask_length, 1).to(interval.device)

            # Ensure dtype consistency
            if interval.dtype in [torch.int, torch.long]:
                noise = noise.round().clamp(interval.min(), interval.max()).to(interval.dtype)

            masked_interval[mask_start:mask_end] = noise

            return masked_interval, interval[mask_start:mask_end]

        except IndexError as e:
            print(f"[ERROR] Index {idx} out of bounds!")
            
            # Get track information safely
            if self.track_ranges:
                total_tracks = len(self.track_ranges)
                track_lengths = [end - start for start, end in self.track_ranges]  # Compute lengths

                print(f"Total Tracks: {total_tracks}")
                print(f"Dataset Length: {sum(track_lengths)} samples")
                
                if idx < total_tracks:
                    print(f"Track {idx} Length: {track_lengths[idx]}")
                else:
                    print("Requested index exceeds available tracks.")

            raise e  # Re-raise the exception after logging

        return masked_interval, interval[mask_start:mask_end]

