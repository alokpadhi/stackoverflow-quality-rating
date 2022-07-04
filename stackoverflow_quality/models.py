from argparse import Namespace
from distutils.util import strtobool
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np

from stackoverflow_quality.data import Tokenizer


class GRU(nn.Module):
    def __init__(self, embedding_dim, vocab_size, rnn_hidden_dim, hidden_dim, dropout, 
                 num_classes, pretrained_embeddings=None, freeze_embeddings=False, padding_idx=0):
        super(GRU, self).__init__()

        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size, padding_idx=padding_idx
            )
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size,
                padding_idx=padding_idx, _weight=pretrained_embeddings
            )

        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

        self.gru = nn.GRU(embedding_dim, rnn_hidden_dim,
                         batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, inputs):
        x_in, seq_lens = inputs
        x_in = self.embeddings(x_in)
        # print(seq_lens.shape)
        packed_in = pack_padded_sequence(x_in, seq_lens.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed_in)
        
        z = torch.cat([h_n[0, :, :], h_n[1, :,:]], dim=1)
        
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)
        
        return z


def load_glove_embeddings(embedding_file: Path=Path("embeddings/glove.6B.100d.txt")) -> Dict:
    """Read the glove embedding file and return the embedding dict.

    Args:
        embedding_file (Path): Location of the glvoe embedding file. Choose the file based on your selected embedding dim.

    Returns:
        Dict: Embedding array for each word
    """
    embeddings = {}
    with open(embedding_file, "r") as fp:
        for index, line in enumerate(fp):
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings


def make_embedding_matrix(embeddings: Dict, word_index: Dict, embedding_dim: int=100):
    """Create the embedding matrix

    Args:
        embeddings (Dict): loaded embedding dictionary from glove file
        word_index (Dict): Word to index dict created from Tokenizer
        embedding_dim (int): Embedding dim. Defaults to 100.

    Returns:
        np.ndarray: Embedding matrix, for each word with given embedding dim
    """
    embedding_matrix = np.zeros((len(word_index), embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def initialize_model(params: Namespace, tokenizer: Tokenizer, num_classes: int, device: torch.device=torch.device("cpu")) -> nn.Module:
    """Initialize the model with all given parameters

    Args:
        params (Namespace): Parameters from params.json
        tokenizer (Tokenizer): Tokenizer class
        num_classes (int): number of target classes. Value equals to len(label_encoder)
        device (torch.device, optional): Device you want to load your model into. Defaults to torch.device("cpu").

    Returns:
        nn.Module: Model initialized
    """
    EMBEDDING_DIM = params.embedding_dim
    RNN_HIDDEN_DIM = params.rnn_hidden_dim
    HIDDEN_DIM = params.hidden_dim
    DROPOUT_P = params.dropout
    PRETRAINED_EMBEDDINGS_TYPE = params.embedding_type
    FREEZE_EMBEDDINGS_TYPE = bool(strtobool(str(params.freeze_embedding_or_not)))
    NUM_CLASSES = num_classes
    VOCAB_SIZE = len(tokenizer)
    

    if PRETRAINED_EMBEDDINGS_TYPE == None and FREEZE_EMBEDDINGS_TYPE == False:
        PRETRAINED_EMBEDDINGS = None
        FREEZE_EMBEDDINGS = False
    elif PRETRAINED_EMBEDDINGS_TYPE == "glove" and FREEZE_EMBEDDINGS_TYPE == True:
        embedding_file = params.embedding_fp
        glove_embeddings = load_glove_embeddings(embedding_file=embedding_file)
        embedding_matrix = make_embedding_matrix(embeddings=glove_embeddings, word_index=tokenizer.token_to_index,
                                        embedding_dim=EMBEDDING_DIM)
        PRETRAINED_EMBEDDINGS = embedding_matrix
        FREEZE_EMBEDDINGS = True

    elif PRETRAINED_EMBEDDINGS_TYPE == "glove" and FREEZE_EMBEDDINGS_TYPE == False: 
        embedding_file = params.embedding_fp
        glove_embeddings = load_glove_embeddings(embedding_file=embedding_file)
        embedding_matrix = make_embedding_matrix(embeddings=glove_embeddings, word_index=tokenizer.token_to_index,
                                        embedding_dim=EMBEDDING_DIM)
        PRETRAINED_EMBEDDINGS = embedding_matrix
        FREEZE_EMBEDDINGS = False

    model = GRU(embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE,
                rnn_hidden_dim=RNN_HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
                dropout=DROPOUT_P, num_classes=NUM_CLASSES, pretrained_embeddings=PRETRAINED_EMBEDDINGS,
                freeze_embeddings=FREEZE_EMBEDDINGS
            )
    model.to(device)

    return model


