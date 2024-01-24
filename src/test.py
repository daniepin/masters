import os
import torch
import wandb
from pathlib import Path

home = Path.home().as_posix()


artifact = wandb.use_artifact('entity/your-project-name/model:v0', type='model')
artifact_dir = artifact.download()
torch.load()
