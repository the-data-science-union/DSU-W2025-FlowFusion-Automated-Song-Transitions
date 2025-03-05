import argparse
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

def main(input_file):
    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(6.0)

    wav, sr = torchaudio.load(input_file)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = (torch.cat([encoded[0] for encoded in encoded_frames], dim=-1))  # [B, n_q, T]
    
    """mean = codes.mean(dim=-1, keepdim=True).float()
    std = codes.std(dim=-1, keepdim=True).float()
    codes = ((codes - mean) / (2*(std + 1e-6)))"""
    
    output_file = "data/processed-tokens/" + input_file.split("/")[-1].split(".")[0] + "_encoded_codes.pt"
    torch.save(codes, output_file)
    print(f"Encoded codes saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode audio file and save the discrete codes with L2 normalization.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file")
    
    args = parser.parse_args()
    main(args.input_file)