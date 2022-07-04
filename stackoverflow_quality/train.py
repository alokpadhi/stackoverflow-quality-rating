from argparse import Namespace
import json
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.optim import Adam

import wandb
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from stackoverflow_quality.data import split_data, prepare, prepare_data_for_training
from config.config import logger
from stackoverflow_quality import models, utils
from config import config


class Trainer(object):
    def __init__(self, model, device, loss_fn=None, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_step(self, dataloader):
        self.model.train()
        loss = 0.0
        
        # iterate over batches
        for i, batch in enumerate(dataloader):
            batch = [item.to(self.device) for item in batch]
            inputs, targets = batch[:-1], batch[-1]
            self.optimizer.zero_grad()
            z = self.model(inputs)
            J = self.loss_fn(z, targets)
            J.backward()
            self.optimizer.step()
            
            loss += (J.detach().item() - loss) / (i+1)
        return loss
    
    def eval_step(self, dataloader):
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []
        
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                batch = [item.to(self.device) for item in batch]
                inputs, y_true = batch[:-1], batch[-1]
                
                z = self.model(inputs)
                J = self.loss_fn(z, y_true).item()
                
                loss += (J - loss) / (i + 1)
                
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())
                
            
            return loss, np.vstack(y_trues), np.vstack(y_probs)
        
    def predict_step(self, dataloader):
        self.model.eval()
        y_probs = []
        
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                batch = [item.to(self.device) for item in batch]
                inputs, targets = batch[:-1], batch[-1]
                
                z = self.model(inputs)
                
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)

            return np.vstack(y_probs)
    
    def train(self, num_epochs, patience, train_dataloader, val_dataloader, wandb_run):
        best_val_loss = np.inf
        for epoch in range(num_epochs):
            print(f"[INFO] Epoch: {epoch+1} training started")
            train_loss = self.train_step(dataloader=train_dataloader)
            print(f"[INFO] Epoch: {epoch+1} training finished")
            print(f"[INFO] Epoch: {epoch+1} evaluation started")
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            print(f"[INFO] Epoch: {epoch+1} evaluation finished")
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                wandb_run.summary["Best val loss"] = best_val_loss
                _patience = patience
            else:
                _patience -= 1
            
            if not _patience:
                print("Stopping Early")
                break
            print(f"[INFO] Logging wandb")
            wandb.log({"train_loss": train_loss})
            wandb.log({"valid_loss": val_loss})
            
            print(
                f"Epoch: {epoch+1}\n"
                f"\t train_loss: {train_loss:.3f}, "
                f"\t val_loss: {val_loss: .3f}, "
                f"\t LR: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )
        return best_model


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List) -> Dict:
    """ Get the performance metrics on test set.

    Args:
        y_true (np.ndarray): Test set true target values
        y_pred (np.ndarray): Predicted target values
        classes (List): Class indentifiers

    Returns:
        Dict: complete performance dictionary
    """
    performance = {"overall": {}, "class": {}}
    
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    performance["overall"]["precision"] = metrics[0]
    performance["overall"]["recall"] = metrics[1]
    performance["overall"]["f1-score"] = metrics[2]
    performance["overall"]["num_samples"] = np.float64(len(y_true))
    
    # Per-class performance
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        performance["class"][classes[i]] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1-score": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }
        
    return performance


def train_the_model(data_path: Path, params: Dict, run) -> Dict:
    """Train the model

    Args:
        data_path (Path): Training data path
        params (Dict): Parameter json file path
        run: wandb run

    Returns:
        Dict: Complete performance dictionary
    """
    # Feature and target
    X, y = prepare(datapath=data_path)
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = split_data(X,y, "label_encoder.json")
    logger.info("Dataset processed and splitted into train/val/test sets.")
    
    train_dataloader, val_dataloader, test_dataloader, tokenizer = prepare_data_for_training(
                                                                    X_train, y_train, X_val, y_val, X_test, y_test)
    logger.info("Train/Val/Test Dataloader and Tokenizer creation completed.")

    device = utils.set_device()

    model = models.initialize_model(params, tokenizer, params.num_classes, device)
    
    loss_fn = nn.CrossEntropyLoss()
    LEARNING_RATE = params.learning_rate
    PATIENCE = params.patience
    NUM_EPOCHS = params.num_epochs

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3)

    trainer = Trainer(model=model, device=device, loss_fn=loss_fn,
                 optimizer=optimizer, scheduler=scheduler)
    
    best_model = trainer.train(NUM_EPOCHS, PATIENCE, train_dataloader, val_dataloader, run)

    test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
    y_pred = np.argmax(y_prob, axis=1)

    performance = get_metrics(y_true, y_pred, classes=label_encoder.classes)

    label_encoder.save(Path(config.ARTIFACTS, "label_encoder_rnn.json"))
    tokenizer.save(Path(config.ARTIFACTS, "tokenizer_rnn.json"))

    label_encoder_artifact = wandb.Artifact("labelencoder", type="feature_engineering", 
                                            metadata=dict(lower=params.lower, stem=params.stem, num_classes=params.num_classes))
    label_encoder_artifact.add_file(Path(config.ARTIFACTS, "label_encoder_rnn.json"))

    tokenizer_artifact = wandb.Artifact("tokenizer", type="feature_engineering",
                                        metadata=dict(num_tokens=5000))
    tokenizer_artifact.add_file(Path(config.ARTIFACTS, "tokenizer_rnn.json"))

    torch.save(best_model.state_dict(), Path(config.ARTIFACTS, "rnn_model.pt"))

    model_artifact = wandb.Artifact("rnn_model", type="model")
    model_artifact.add_file(Path(config.ARTIFACTS, "rnn_model.pt"))

    with open(Path(config.ARTIFACTS, "performance.json"), "w") as fp:
        json.dump(performance, indent=2, sort_keys=False, fp=fp)

    performance_artifact = wandb.Artifact("performance", type="evaluation")
    performance_artifact.add_file(Path(config.ARTIFACTS, "performance.json"))

    run.log_artifact(label_encoder_artifact)
    run.log_artifact(tokenizer_artifact)
    run.log_artifact(model_artifact)
    run.log_artifact(performance_artifact)

    run.log({"precision": performance["overall"]["precision"], 
            "recall": performance["overall"]["recall"], 
            "f1-score": performance["overall"]["f1-score"]})

    run.log({"complete_performance": performance})
    
    return performance

    

    

    



