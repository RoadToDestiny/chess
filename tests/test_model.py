"""
Tests for model training, probabilistic metrics, and serialization.
"""

from pathlib import Path
import numpy as np
import pytest

from chess_error_prediction.model.metrics import compute_classification_metrics
from chess_error_prediction.model.train import train_xgboost_classifier, save_model
from chess_error_prediction.model.calibration import plot_calibration_curve
from chess_error_prediction.dataset.build_dataset import generate_synthetic_dataset
from chess_error_prediction.dataset.split import split_by_games
from chess_error_prediction.schemas import POSITION_FEATURES


def test_metrics_computation():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    metrics = compute_classification_metrics(y_true, y_prob)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert metrics["roc_auc"] > 0.8
    assert metrics["brier_score"] < 0.2


def test_xgboost_training_and_serialization(tmp_path: Path):
    df = generate_synthetic_dataset(num_games=15, random_seed=42)
    train_df, val_df, _ = split_by_games(df, 0.70, 0.15, 0.15, random_seed=42)

    fast_params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    model, val_metrics = train_xgboost_classifier(
        train_df=train_df,
        val_df=val_df,
        feature_cols=POSITION_FEATURES,
        params=fast_params,
    )

    assert "roc_auc" in val_metrics
    assert 0.0 <= val_metrics["roc_auc"] <= 1.0

    save_dir = tmp_path / "model_out"
    model_path = save_model(model, save_dir, POSITION_FEATURES, fast_params)
    assert model_path.exists()
    assert (save_dir / "model_metadata.json").exists()


def test_calibration_curve_generation(tmp_path: Path):
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.15, 0.65, 0.3, 0.9])

    plot_file = tmp_path / "calibration.png"
    prob_true, prob_pred = plot_calibration_curve(y_true, y_prob, save_path=plot_file)
    assert len(prob_true) == len(prob_pred)
    assert plot_file.exists()
