import os
import torch


# Experiment parameters
ENV_NAME = os.environ.get("ENV_NAME", "halfcheetah-medium-v2")

# Noise
NOISE_DIM = int(os.environ.get("NOISE_DIM", 40))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", 0.5))
NOISE_TYPE = os.environ.get("NOISE_TYPE", "concat")
SEED = int(os.environ.get("SEED", 11))

# Training hyperparameters
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
EPOCHS = int(os.environ.get("EPOCHS", 1))
PRETRAIN_EPOCHS = int(os.environ.get("PRETRAIN_EPOCHS", 1))
PRETRAIN_BS = int(os.environ.get("PRETRAIN_BS", 512))

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")