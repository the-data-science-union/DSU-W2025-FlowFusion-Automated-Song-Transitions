import torch

# Toggle for enabling/disabling W&B logging
WANDB_LOGS = True  

# Define Training Parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 250
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
INTERVAL_LENGTH = 32
MASK_LENGTH = 2
SAMPLE_RATE = 50
FILE_PATH = "/Users/jacobbia/Documents/UCLA/DSU/AIDJ/DSU-W2025-FlowFusion-Automated-Song-Transitions/data/processed-tokens"

# Normalization statistics
EPSILON = 1e-6
VOCAB_SIZE = 1024

# Model Initialization
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 2048
MAX_SEQ_LENGTH = 1600
DROPOUT = 0.2

#Mamba Hyperparams
MAMBA_D_STATE=2
MAMBA_D_CONV = 2
MAMBA_EXPAND = 2