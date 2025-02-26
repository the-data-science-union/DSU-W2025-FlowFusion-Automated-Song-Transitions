import torch 
file_path = '/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/data/processed-tokens/Blinding_Lights_-_The_Weeknd_encoded_codes.pt'
data = torch.load(file_path)  # Load the tokenized input

if not isinstance(data, torch.Tensor):
    raise ValueError(f"Loaded data is not a tensor! Got {type(data)}")

print(f"Loaded data shape: {data.shape}")  # Debugging info