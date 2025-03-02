import argparse
import torch

def normalize_pt_file(input_file):
    # Load the existing .pt file
    codes = torch.load(input_file).to(torch.float32)  # Convert to float for normalization
    
    # Apply L2 normalization across the last dimension
    normalized_codes = torch.nn.functional.normalize(codes, p=2, dim=-1)
    
    # Convert back to long dtype
    normalized_codes = normalized_codes.to(torch.long)
    
    # Save the normalized codes
    output_file = input_file.replace(".pt", "_l2_normalized.pt")
    torch.save(normalized_codes, output_file)
    print(f"Normalized codes saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply L2 normalization to an existing .pt file.")
    parser.add_argument("input_file", type=str, help="Path to the input .pt file")
    
    args = parser.parse_args()
    normalize_pt_file(args.input_file)