import torch

# Set to False to disable W&B logging
WANDB_LOGS = True

#data information
VOCAB_SIZE = 1024
AUDIO_CHANNELS = 4
DATA_PATH = "/home/aditya/DSU-W2025-FlowFusion-Automated-Song-Transitions/data/processed-tokens/"
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
INTERVAL_LENGTH = 32
MASK_LENGTH = 2
SAMPLE_RATE = 50

#architectural params
D_MODEL = 512
NUM_LAYERS = 5
NUM_HEADS = 16
D_FF = 2048
MAX_SEQ_LENGTH = 1600
DROPOUT = 0.2

#hyperparams
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 1e-4
