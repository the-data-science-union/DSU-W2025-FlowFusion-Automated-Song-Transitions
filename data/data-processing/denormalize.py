import torch
import pandas as pd

def denormalize(codes, normalization_info_file="data/normalization_info.csv"):
    normalization_df = pd.read_csv(normalization_info_file)
    
    input_file = "data/processed-tokens/Don't_Stop_The_Music_-_Rihanna_encoded_codes.pt"
    normalization_info = normalization_df[normalization_df['input_file'] == input_file].iloc[0]
    
    mean = torch.tensor(normalization_info['mean'])
    std = torch.tensor(normalization_info['std'])
    
    return codes * std + mean

def main():
    normalized_codes_file = "data/processed-tokens/example_audio_encoded_codes.pt"
    normalized_codes = torch.load(normalized_codes_file)
    
    # Denormalize the codes
    denormalized_codes = denormalize(normalized_codes)
    
    print(f"Denormalized codes: {denormalized_codes}")

if __name__ == "__main__":
    main()
