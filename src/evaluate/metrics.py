from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

def evaluate_binary(model, X, y) -> dict:
    """
    Returns accuracy and log loss on provided set.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=[0,1]))
    }
