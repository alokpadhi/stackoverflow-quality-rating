from argparse import Namespace
import json
import os
from pathlib import Path
from typing import Dict, Optional
import wandb
import torch
from config.config import logger
from config import config
from data import split_data, prepare, prepare_data_for_training
from stackoverflow_quality import utils, train, data, models, predict


def download_data(filename: str="", use_wandb: bool=False) -> None:
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


def train_model(data_path: Path=Path(config.DATA_DIR, "train.csv"), 
                params_fp: Path=Path(config.CONFIG_DIR, "params.json"), 
                run_name: Optional[str]="model",
                project_name: Optional[str]="stackoverflow-quality") -> None:
    """ Train the defined model

    Args:
        data_path (Path, optional): Location of your training data.. Defaults to Path(config.DATA_DIR, "train.csv").
        params_fp (Path, optional): Parameter json file location. Defaults to Path(config.CONFIG_DIR, "params.json").
        run_name (Optional[str], optional): WandB run_name, to log the metrics and artifacts during training. Defaults to "model".
        project_name (Optional[str], optional): WandB project name, where you want to log. Defaults to "stackoverflow-quality".
    """
    params = Namespace(**utils.load_dict(filepath=params_fp))
    MODEL_CONFIG = dict(
    learning_rate=params.learning_rate,
    patience=params.patience,
    epochs=params.num_epochs,
    target_classes=params.num_classes,
    optimizer="Adam",
    scheduler="ReduceLROnPlateau",
    loss_function="Cross Entropy",
    embedding=params.embedding_type,
    embedding_dim=params.embedding_dim,
    model="GRU",
    )

    run = wandb.init(project=project_name, config=MODEL_CONFIG, name=run_name)
    performance = train.train_the_model(data_path, params, run)
    logger.info("Model training finished.")
    logger.info(json.dumps(performance["overall"], indent=2))
    wandb.finish()


def load_artifact(params_fp: Path=Path(config.CONFIG_DIR, "params.json"),
                project_name: Optional[str]="stackoverflow-quality",
                device: torch.device = torch.device("cpu")) -> Dict:
    params = Namespace(**utils.load_dict(params_fp))
    
    run = wandb.init(project=project_name)
    
    if Path(config.WANDB_ARTIFACTS,"label_encoder_rnn.json").exists():
        label_encoder = data.LabelEncoder.load(Path(config.WANDB_ARTIFACTS,"label_encoder_rnn.json"))
    else:
        label_encoder_artifact = run.use_artifact('alokpadhi/test/labelencoder:latest', type='feature_engineering')
        le_artifact_dir =label_encoder_artifact.download(config.WANDB_ARTIFACTS)
        label_encoder = data.LabelEncoder.load(Path(le_artifact_dir, "label_encoder_rnn.json"))

    if Path(config.WANDB_ARTIFACTS, "tokenizer_rnn.json").exists():
        tokenizer = data.Tokenizer.load(Path(config.WANDB_ARTIFACTS, "tokenizer_rnn.json"))
    else:
        tokenizer_artifact = run.use_artifact('alokpadhi/test/tokenizer:latest', type='feature_engineering')
        tokenizer_artifact_dir = tokenizer_artifact.download(config.WANDB_ARTIFACTS)
        tokenizer = data.Tokenizer.load(Path(tokenizer_artifact_dir, "tokenizer_rnn.json"))

    if Path(config.WANDB_ARTIFACTS, "rnn_model.pt").exists():
        model_state = torch.load(Path(config.WANDB_ARTIFACTS, "rnn_model.pt"), map_location=device)
    else:
        model_artifact = run.use_artifact('alokpadhi/test/rnn_model:latest', type='model')
        model_artifact_dir = model_artifact.download(config.WANDB_ARTIFACTS)
        model_state = torch.load(Path(model_artifact_dir, "rnn_model.pt"), map_location=device)

    if Path(config.WANDB_ARTIFACTS, "performance.json").exists():
        performance = utils.load_dict(Path(config.WANDB_ARTIFACTS, "performance.json"))
    else:
        performance_artifact = run.use_artifact('alokpadhi/test/performance:latest', type="evaluation")
        performance_artifact_dir= performance_artifact.download(config.WANDB_ARTIFACTS)
        performance = utils.load_dict(Path(performance_artifact_dir, "performance.json"))

    model = models.initialize_model(params=params, tokenizer=tokenizer, num_classes=len(label_encoder))
    model.load_state_dict(model_state)

    wandb.finish()

    return {
        "params": params,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "performance": performance
    }


def predict_rating(text: str, 
                params_fp: Path=Path(config.CONFIG_DIR, "params.json"),
                project_name: Optional[str]="stackoverflow-quality",
                device: torch.device = torch.device("cpu")) -> Dict:
    """Predict the quality rating for a given input text

    Args:
        text (str): input text
        params_fp (Path, optional): Parameter file location. Defaults to Path(config.CONFIG_DIR, "params.json").
        project_name (Optional[str], optional): WandB project name. Defaults to "stackoverflow-quality".
        device (torch.device, optional): Device type you want to use for prediction. Defaults to torch.device("cpu").
    """
    artifacts = load_artifact(params_fp=params_fp, project_name=project_name, device=device)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))


def get_performance(project_name: Optional[str]="stackoverflow-quality") -> Dict:
    """Get the complete performance detail of your trained model/

    Args:
        project_name (Optional[str], optional): WandB project name. Defaults to "stackoverflow-quality".
    """
    run = wandb.init(project=project_name)

    if Path(config.WANDB_ARTIFACTS, "performance.json").exists():
        performance = utils.load_dict(Path(config.WANDB_ARTIFACTS, "performance.json"))
    else:
        performance_artifact = run.use_artifact('alokpadhi/test/performance:latest', type="evaluation")
        performance_artifact_dir= performance_artifact.download(config.WANDB_ARTIFACTS)
        performance = utils.load_dict(Path(performance_artifact_dir, "performance.json"))
    wandb.finish()
    
    logger.info(json.dumps(performance, indent=2))