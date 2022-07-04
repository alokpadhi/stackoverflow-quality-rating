from distutils.util import strtobool
from typing import List, Dict

import numpy as np
import torch

from stackoverflow_quality import data, train, preprocess


def predict(texts: List, artifacts: Dict, device: torch.device = torch.device("cpu")) -> Dict:
    """Predict the rating quality of the given list of stackoverflow texts

    Args:
        texts (List): Input text as in stackoverflow body or title contents
        artifacts (Dict): Load necessary artifacts like label_encoder, tokenizer, model etc
        device (torch.device, optional): Device type you want to run your predictions. Defaults to torch.device("cpu").

    Returns:
        Dict: Predicted quality rating for each input text in a dictionary format.
    """
    params = artifacts["params"]
    label_encoder = artifacts["label_encoder"]
    tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]

    preprocessed_texts = [preprocess.preprocess(
        text,
        lower=bool(strtobool(str(params.lower))),
        stem=bool(strtobool(str(params.stem))),
        )
        for text in texts
    ]

    X = np.array(tokenizer.texts_to_sequences(preprocessed_texts), dtype="object")
    y_filler = label_encoder.encode([label_encoder.classes[0]] * len(X))
    dataset = data.Dataset(X=X, y=y_filler)
    dataloader = dataset.create_dataloader(batch_size=int(params.batch_size))

    trainer = train.Trainer(model=model, device=device)
    y_prob = trainer.predict_step(dataloader)
    y_pred = np.argmax(y_prob, axis=1)
    quality_ratings = label_encoder.decode(y_pred)

    predictions = [
        {
            "input_text": texts[i],
            "predicted_quality_rating": quality_ratings[i]
        }
        for i in range(len(quality_ratings))
    ]

    return predictions