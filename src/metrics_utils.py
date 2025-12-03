from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", pos_label=1),
        "recall": recall_score(y_true, y_pred, average="binary", pos_label=1),
        "f1": f1_score(y_true, y_pred, average="binary", pos_label=1),
    }
    return metrics
