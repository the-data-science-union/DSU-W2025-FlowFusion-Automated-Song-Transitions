import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch.utils.data import DataLoader
from mamba_models.bidrectional_mamba import BidirectionalMamba
from data.data_loader import MusicDataset
import os

# Define parameters matching the training script
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INTERVAL_LENGTH = 32
MASK_LENGTH = 2
SAMPLE_RATE = 50
FILE_PATH = "/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/data/processed-tokens/"
VOCAB_SIZE = 1024
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 2048
MAX_SEQ_LENGTH = 1600
DROPOUT = 0.2
DEVICE = "cuda:2"

def convert_to_wav(denorm_data, output_file, device=DEVICE):  # Add device parameter
    model = EncodecModel.encodec_model_48khz().to(device)  # Move model to specified device
    model.set_target_bandwidth(6.0)
    
    # Expected shape: [B, N, T], where N is number of codebooks (e.g., 4), T is time
    if denorm_data.dim() == 2:  # [T, N] from dataset
        denorm_data = denorm_data.transpose(0, 1)  # [N, T]
        denorm_data = denorm_data.unsqueeze(0)  # [1, N, T]
    elif denorm_data.dim() == 3:  # [1, T, N]
        denorm_data = denorm_data.transpose(1, 2)  # [1, N, T]
    
    denorm_data = denorm_data.long().to(device)  # Ensure integer codes and move to device
    
    print(f"Denorm data shape: {denorm_data.shape}")  # Debug print
    
    encoded_frame = (denorm_data, None)  # (codes, scale) tuple
    with torch.no_grad():
        decoded_audio = model.decode([encoded_frame])
    
    # Move decoded audio back to CPU for saving
    torchaudio.save(output_file, decoded_audio.squeeze(0).cpu(), sample_rate=model.sample_rate)
    print(f"Saved WAV file: {output_file}")

def test_model(model_path, data_path, device, sample_number):
    # Initialize the model with same parameters as in training
    model = BidirectionalMamba(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
        device=device  # Use the passed device parameter
    ).to(device)
    
    # Load the saved model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    # Initialize dataset
    test_dataset = MusicDataset(
        data_path,
        INTERVAL_LENGTH,
        MASK_LENGTH,
        SAMPLE_RATE
    )
    
    # Get the first item from the dataset
    masked_intervals, original_masks = test_dataset[sample_number]
    
    # Add batch dimension since we're processing single item
    masked_intervals = masked_intervals.unsqueeze(0).to(device)
    original_masks = original_masks.unsqueeze(0).to(device)
    
    # Get model predictions
    with torch.no_grad():  # No gradient calculation needed for inference
        outputs = model(masked_intervals)  # [batch, seq_len, num_codebooks, vocab_size] = [1, 1600, 4, 1024]
        
        # Extract the mask region as in training
        mask_start = (outputs.shape[1] - original_masks.shape[1]) // 2
        mask_end = mask_start + original_masks.shape[1]
        predicted_masks = outputs[:, mask_start:mask_end, :, :]  # [1, 100, 4, 1024]
    
    # Convert probability distributions to indices by taking argmax along the last dimension
    outputs_indices = torch.argmax(outputs, dim=-1)  # [1, 1600, 4]
    predicted_masks_indices = torch.argmax(predicted_masks, dim=-1)  # [1, 100, 4]
    
    # Print shapes and results
    print("Input masked_intervals shape:", masked_intervals.shape)
    print("Original masks shape:", original_masks.shape)
    print("Model output shape:", outputs.shape)
    print("Predicted masks shape:", predicted_masks.shape)
    print("Model output indices shape:", outputs_indices.shape)
    print("Predicted masks indices shape:", predicted_masks_indices.shape)
    
    print("\nSample of predicted mask indices:", predicted_masks_indices[0][:10])  # First 10 predictions
    print("Sample of original mask values:", original_masks[0][:10])  # First 10 original values
    
    return predicted_masks_indices, original_masks  # Return the indices instead of probabilities


def main():
    # Define paths
    model_path = "/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/bidirectional_mamba_epoch_250.pt"
    data_path = FILE_PATH
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    if not os.path.exists(data_path):
        raise ValueError(f"Data path not found at: {data_path}")
    
    # Run test
    predicted_masks, original_masks = test_model(model_path, data_path, DEVICE, 500)

    convert_to_wav(predicted_masks, "sample_prediction.wav")
    convert_to_wav(original_masks, "original_mask.wav")

if __name__ == "__main__":
    main()