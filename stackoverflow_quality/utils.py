import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seeds(seed=1234):
    """set seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-gpu


def set_device(cuda=False):
    """ Set whether you want to use cuda or not.

    Args:
        cuda (bool, optional): True if you want to use cuda else False. Defaults to False.

    Returns:
        torch.device: Return torch.device. cuda or cpu/ 
    """
    device = torch.device("cuda" if(
                        torch.cuda.is_available() and cuda) else "cpu")
    
    torch.set_default_tensor_type("torch.FloatTensor")

    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    return device