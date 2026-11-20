"""
Metrics calculation for chess blunder probability prediction.
Evaluates probabilistic forecasts: ROC-AUC, PR-AUC, Brier Score, Log Loss.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes standard probabilistic and binary metrics for error classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities of blunder (class 1).
    threshold : float
        Decision threshold for binary metrics (default 0.5).

    Returns
    -------
    Dict[str, float]
        Dictionary containing roc_auc, pr_auc, brier_score, log_loss,
        accuracy, precision, recall, f1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    # Check if both classes are present
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier_score": float(brier_score_loss(y_true, y_prob_clipped)),
            "log_loss": float(log_loss(y_true, y_prob_clipped)),
            "accuracy": float(accuracy_score(y_true, (y_prob >= threshold).astype(int))),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob_clipped))

    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }
