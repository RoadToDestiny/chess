"""
Unit and integration tests for Stage 8: Model Training and Baseline Evaluation.
"""

import numpy as np
import pandas as pd
import pytest
from chess_error_prediction.schemas import ALL_FEATURES
from chess_error_prediction.model import (
    train_models,
    compute_classification_metrics,
    create_logistic_regression_pipeline,
)


@pytest.fixture
def synthetic_split_data():
    """Generates synthetic tabular datasets with valid features and target."""
    np.random.seed(42)
    n_train = 200
    n_val = 50
    n_test = 50

    def make_df(n, start_game_id):
        data = {f: np.random.randn(n).astype(np.float32) for f in ALL_FEATURES}
        data["move_number"] = np.random.randint(1, 60, size=n)
        data["game_phase"] = np.random.choice([1, 2, 3], size=n)
        data["has_prev_move"] = np.random.choice([0, 1], size=n)
        data["game_id"] = [f"game_{start_game_id + (i // 10)}" for i in range(n)]
        # Imbalanced binary target (~15% blunder rate)
        data["is_error"] = (np.random.rand(n) > 0.85).astype(int)
        return pd.DataFrame(data)

    return make_df(n_train, 0), make_df(n_val, 100), make_df(n_test, 200)


def test_compute_classification_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.9, 0.2])
    metrics = compute_classification_metrics(y_true, y_prob)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["brier_score"] >= 0.0


def test_train_models_end_to_end(synthetic_split_data):
    train_df, val_df, test_df = synthetic_split_data
    res = train_models(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=ALL_FEATURES,
        target_col="is_error",
        group_col="game_id",
        enable_calibration=True,
        device="cpu",  # run test safely on CPU
    )

    assert res.xgb_model is not None
    assert res.lr_pipeline is not None
    assert "xgboost" in res.metrics_val
    assert "logistic_regression" in res.metrics_val
    assert "xgboost" in res.metrics_test
    assert "logistic_regression" in res.metrics_test

    # Test predict probabilities range
    test_probs = res.xgb_model.predict_proba(test_df[ALL_FEATURES])[:, 1]
    assert np.all((test_probs >= 0.0) & (test_probs <= 1.0))

    lr_probs = res.lr_pipeline.predict_proba(test_df[ALL_FEATURES])[:, 1]
    assert np.all((lr_probs >= 0.0) & (lr_probs <= 1.0))
