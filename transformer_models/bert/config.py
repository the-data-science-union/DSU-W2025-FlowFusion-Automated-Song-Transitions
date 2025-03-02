import torch

#data information
vocab_size = 32759
data_path = "/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/data/processed-tokens/"
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

#architectural params
d_model = 256
num_layers = 12
num_heads = 8
d_ff = 512
max_seq_length = 1600

#hyperparams
batch_size = 16
num_epochs = 10
learning_rate = 1e-4