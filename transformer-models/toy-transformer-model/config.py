import torch
import os

BASE_DIR = '/scratch/aditya'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Ensure these directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Other config parameters...
TRAIN_FILE = os.path.join(DATA_DIR, 'corpus.txt')

# Model Architecture
VOCAB_SIZE = 30522  # Default for BERT-base
D_MODEL = 768
NUM_LAYERS = 12
NUM_HEADS = 12
D_FF = 3072
MAX_LENGTH = 128  # Reduced from 512 to 128 as Brown corpus sentences are typically shorter
DROPOUT = 0.1

# Training Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Reduced from 40 to 10 as we're using a smaller dataset
LEARNING_RATE = 5e-5  # Reduced from 1e-4 to 5e-5 for fine-tuning
WARMUP_STEPS = 1000  # Reduced from 10000 to 1000 due to smaller dataset

# MLM Task
MLM_PROBABILITY = 0.15

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging and Saving
LOG_INTERVAL = 100
SAVE_INTERVAL = 1000  # Reduced from 10000 to 1000 due to smaller dataset

# Optimizer
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0

# Miscellaneous
SEED = 42