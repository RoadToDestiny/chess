"""Group-aware out-of-fold probability calibration."""

from dataclasses import dataclass
from typing import Callable, Dict, Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class GroupedCalibrationResult:
    """Fitted model and calibrators trained without using the test split."""

    estimator: Any
    platt: LogisticRegression
    isotonic: IsotonicRegression
    oof_probability: np.ndarray
    fold_count: int


def fit_grouped_calibrators(
    X: Any,
    y: Any,
    groups: Any,
    estimator_factory: Callable[[], Any],
    n_splits: int = 5,
) -> GroupedCalibrationResult:
    """Fit base and calibration models using grouped out-of-fold predictions.

    The base estimator is fitted on each training fold, while both calibrators
    are fitted only on out-of-fold predictions from the training data.
    """
    group_values = np.asarray(groups)
    target_values = np.asarray(y)
    if np.unique(target_values).size < 2:
        raise ValueError("Both target classes are required for calibration.")
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 2:
        raise ValueError("At least two distinct groups are required for calibration.")
    fold_count = min(int(n_splits), len(unique_groups))
    if fold_count < 2:
        raise ValueError("n_splits must produce at least two grouped folds.")

    splitter = GroupKFold(n_splits=fold_count)
    oof_probability = np.full(len(y), np.nan, dtype=float)
    for train_idx, valid_idx in splitter.split(X, y, group_values):
        estimator = estimator_factory()
        estimator.fit(
            X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx],
            target_values[train_idx],
        )
        valid_X = X.iloc[valid_idx] if hasattr(X, "iloc") else X[valid_idx]
        oof_probability[valid_idx] = estimator.predict_proba(valid_X)[:, 1]

    if np.isnan(oof_probability).any():
        raise RuntimeError("Grouped OOF calibration produced incomplete predictions.")

    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(oof_probability.reshape(-1, 1), target_values)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(oof_probability, target_values)

    estimator = estimator_factory()
    estimator.fit(X, target_values)
    return GroupedCalibrationResult(
        estimator=estimator,
        platt=platt,
        isotonic=isotonic,
        oof_probability=oof_probability,
        fold_count=fold_count,
    )


def evaluate_calibrators(
    result: GroupedCalibrationResult,
    X: Any,
    y: Any,
) -> Dict[str, Dict[str, float]]:
    """Evaluate uncalibrated, Platt, and isotonic probabilities."""
    raw = result.estimator.predict_proba(X)[:, 1]
    platt = result.platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    isotonic = np.asarray(result.isotonic.predict(raw), dtype=float)

    return {
        "uncalibrated": _probability_metrics(y, raw),
        "platt": _probability_metrics(y, platt),
        "isotonic": _probability_metrics(y, isotonic),
    }


def _probability_metrics(y: Any, probability: np.ndarray) -> Dict[str, float]:
    return {
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }
