from typing import Dict

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def classification_metrics(targets, predictions, classes: int) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    matrix = confusion_matrix(targets, predictions, labels=np.arange(classes))
    support = matrix.sum(axis=1)
    per_class = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(classes, dtype=np.float64),
        where=support > 0,
    )
    return {
        "oa": float(np.mean(targets == predictions)),
        "aa": float(per_class.mean()),
        "kappa": float(cohen_kappa_score(targets, predictions)),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
