from pathlib import Path
from typing import Optional

import math
import re
import string
import json

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split

import collections
import itertools
from collections import Counter


import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import preprocess
from config import config
from config.config import logger


class LabelEncoder(object):
    """Encode labels into unqiue ids/integers"""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v:k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        
    def __len__(self):
        return len(self.class_to_index)
    
    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"
    
    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v:k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self
    
    def encode(self, y):
        encoded = np.zeros(len(y), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded
    
    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
            
        return classes
    
    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)
    
    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def prepare(datapath: str):
    """

    Args:
        datapath (str): _description_
    """
    logger.info(f"Reading {datapath} to a pandas dataframe.")
    train_df = pd.read_csv(datapath)
    train_df['text'] = train_df['Title'] + train_df['Body']
    train_df = train_df.drop(['Title', 'Body'], axis=1)
    logger.info(f"Text preprocessing ...")
    train_df.text = train_df.text.apply(preprocess, lower=True, stem=False)
    X = train_df.text.to_numpy()
    y = train_df.Y

    return X, y


def split_data(X, y, labelencoder_file: str=None, train_size: float=0.7, val_size: float=0.15, test_size: float=0.15):
    """ Read the raw unprocessed data, process it, encode the labels, and then split into train, val and test sets.

    Args:
        datapath (str): Location of your unprocessed raw data.
        labelencoder_file (Optional[None]): If you have already saved your LabelEncoder Json file, then provide the location here.
        train_size (float, optional): Training set split size. Defaults to 0.7.
        val_size (float, optional): Validation split size. Defaults to 0.15.
        test_size (float, optional): Test split size. Defaults to 0.15.

    Returns:
        Returns split datasets.
    """
    if labelencoder_file:
        try:
            label_encoder = LabelEncoder.load(Path(config.ARTIFACTS, labelencoder_file))
            logger.info(f"{label_encoder} Loaded")
        except Exception as e:
            logger.error(e)
    else:
        try:
            label_encoder = LabelEncoder()
            label_encoder.fit(y)
            label_encoder.save(Path(config.ARTIFACTS, "label_encoder.json"))
            logger.info("LabelEncoder fit finished and saved.")
        except Exception as e:
            logger.error(e)

    # split sizes
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    # To ensure the target distribution remains same across the splits
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, 
                                                stratify=y)

    # split for validation and test set
    X_val, X_test, y_val, y_test = train_test_split(
                                X_, y_, train_size=0.5, stratify=y_)

    # Encode all our labels
    y_train = label_encoder.encode(y_train)

    y_val = label_encoder.encode(y_val)

    y_test = label_encoder.encode(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Tokenizer
class Tokenizer(object):
    def __init__(self, char_level, num_tokens=None, pad_token="<PAD>",
                oov_token="<UNK", token_to_index=None):
        self.char_level = char_level
        self.separator = "" if self.char_level else " "
        if num_tokens: num_tokens -=2 # pad token and unk token excluded
        self.num_tokens = num_tokens
        self.pad_token = pad_token
        self.oov_token = oov_token
        
        if not token_to_index:
            token_to_index = {pad_token: 0, oov_token: 1}
        self.token_to_index = token_to_index
        self.index_to_token = {v:k for k, v in self.token_to_index.items()}
        
    def __len__(self):
        return len(self.token_to_index)
    
    def __str__(self):
        return f"<Tokenizer(num_tokens={len(self)})>"
    
    def fit_on_texts(self, texts):
        if not self.char_level:
            texts = [text.split(' ') for text in texts]
        all_tokens = [token for text in texts for token in text]
        counts = Counter(all_tokens).most_common(self.num_tokens)
        self.min_token_freq = counts[-1][1]
        
        for token, count in counts:
            index = len(self)
            self.token_to_index[token] = index
            self.index_to_token[index] = token
        return self
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            if not self.char_level:
                text = text.split(" ")
            sequence = []
            for token in text:
                sequence.append(self.token_to_index.get(
                    token, self.token_to_index[self.oov_token]
                ))
            sequences.append(sequence)
        return sequences
            
    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = []
            for index in sequence:
                text.append(self.index_to_token.get(
                    index, self.oov_token
                ))
            texts.append(self.separator.join([token for token in text]))
        return texts
    
    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {
                "char_level": self.char_level,
                "oov_token": self.oov_token,
                "token_to_index": self.token_to_index
            }
            json.dump(contents, fp, indent=4, sort_keys=False)
    
    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


# Pad the sequences to max_length
def pad_sequences(sequences, max_seq_len=0):
    """Pad sequences to max_length in sequence"""
    # max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))
    # num_classes = sequences[0].shape[-1]
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        if len(sequence) < max_seq_len:
            padded_sequences[i][:len(sequence)] = sequence
        else:
            # print(sequence)
            padded_sequences[i][:max_seq_len] = sequence[:200]
    return padded_sequences


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return [X, len(X), y]
    
    def collate_fn(self, batch):
        batch = np.array(batch)
        X = batch[:, 0]
        seq_lens = batch[:, 1]
        y = batch[:, 2]
        
        X = pad_sequences(X, 200)
        
        seq_lens = np.array([seq_len if seq_len < 200 else 200 for seq_len in seq_lens])
        X = torch.LongTensor(X.astype(np.int32))
        seq_lens = torch.LongTensor(seq_lens.astype(np.int32))
        y = torch.LongTensor(y.astype(np.int32))
        
        return X, seq_lens, y
    
    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self, batch_size=batch_size, collate_fn=self.collate_fn,
            shuffle=shuffle, drop_last=drop_last, pin_memory=True
        )


def prepare_data_for_training(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int=32, ):
    tokenizer = Tokenizer(char_level=False, num_tokens=5000)
    tokenizer.fit_on_texts(texts=X_train)
    VOCAB_SIZE = len(tokenizer)

    # convert texts to sequences
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(X_test)

    train_dataset = Dataset(X=X_train, y=y_train)
    val_dataset = Dataset(X=X_val, y=y_val)
    test_dataset = Dataset(X=X_test, y=y_test)

    train_dataloader = train_dataset.create_dataloader(batch_size=batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=batch_size)
    test_dataloader = test_dataset.create_dataloader(batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader