import os
from pathlib import Path
from typing import Optional
import wandb
from config.config import logger
from config import config
from data import split_data, prepare, prepare_data_for_training

def download_data(filename: str="", use_wandb: bool=False):
    """Collect the raw data either from the given file path or download directly from wandb.

    Args:
        filename (str): Your local file path for the raw unprocessed dataset
        use_wandb (bool, optional): Use wandb to download the raw data. Defaults to False.
    """
    if use_wandb:
        run = wandb.init()
        artifact = run.use_artifact('alokpadhi/stackoverflow-quality/raw_dataset:v0', type='raw_data')
        artifact_dir = artifact.download(Path(config.DATA_DIR))
        logger.info("Data downloaded from wandb.")
    else:
        if os.path.isfile(Path(config.DATA_DIR, filename)):
            logger.info("File already exists; Ready to use.")
        else:
            logger.error("File does not exist.")


def prepare_data(data_path: str):
    # Feature and target
    X, y = prepare(datapath=data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X,y, "label_encoder.json")
    logger.info("Dataset processed and splitted into train/val/test sets.")
    
    prepare_data_for_training(X_train, y_train, X_val, y_val, X_test, y_test)
    logger.info("Train/Val/Test Dataloader creation completed.")


def train_model():
    pass


def predict_rating():
    pass


def get_performance():
    pass