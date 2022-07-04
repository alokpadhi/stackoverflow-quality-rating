import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List
from config.config import STOPWORDS


porter = PorterStemmer()

def preprocess(text: pd.Series, lower: bool = True, stem: bool = False, 
                filters: List ="[!\"'#$%&()*\+,-.:;<=>?@\\\[\]^_`{|}~]",
                stopwords=STOPWORDS):
    """preprocess the required column

    Args:
        text (_type_, optional): Tetx column data. Defaults to True.stem: bool = False, filters: List ="[!\"'#$%&()*\+,-.:;<=>?@\\[\]^_`{|}~]".
        stopwords (_type_, optional): _description_. Defaults to STOPWORDS.
    """
    # lower the text
    if lower:
        text = text.lower()
    
    # remove the stopwords
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)
    
    # remove <p> and </p> tags
    text = re.sub(r"[^(a-zA-Z0-9)\s]", " ", text)
    text = re.sub(r"\bp\b","", text)
    
    # spacing and filters
    # text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    # text = re.sub(filters, r"", text)
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()
    
    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        text = " ".join([porter.stem(word) for word in text.split(" ")])

    return text