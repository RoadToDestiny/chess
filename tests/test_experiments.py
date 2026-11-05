"""
Tests for scientific experiment execution modules (Main, Exp 1, Exp 2).
"""

from pathlib import Path
import pandas as pd

from chess_error_prediction.dataset.build_dataset import generate_synthetic_dataset
from chess_error_prediction.dataset.split import split_by_games
from chess_error_prediction.experiments.run_main import run_main_experiment
from chess_error_prediction.experiments.exp1_feature_sets import run_feature_sets_experiment
from chess_error_prediction.experiments.exp2_thresholds import run_threshold_sensitivity_experiment


def test_main_experiment_execution(tmp_path: Path):
    df = generate_synthetic_dataset(num_games=15, random_seed=42)
    train_df, val_df, test_df = split_by_games(df, 0.70, 0.15, 0.15, random_seed=42)

    fast_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    metrics = run_main_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        model_params=fast_params,
    )

    assert "roc_auc" in metrics
    assert (tmp_path / "artifacts" / "models" / "model.json").exists()
    assert (tmp_path / "artifacts" / "runs" / "main_run.json").exists()
    assert (tmp_path / "reports" / "tables" / "main_experiment_metrics.csv").exists()
    assert (tmp_path / "reports" / "figures" / "calibration_curve.png").exists()


def test_feature_sets_experiment(tmp_path: Path):
    df = generate_synthetic_dataset(num_games=15, random_seed=42)
    train_df, val_df, test_df = split_by_games(df, 0.70, 0.15, 0.15, random_seed=42)

    fast_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    comp_df = run_feature_sets_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        reports_dir=tmp_path / "reports",
        model_params=fast_params,
    )

    assert len(comp_df) == 2
    assert "delta_roc_auc_vs_baseline" in comp_df.columns
    assert (tmp_path / "reports" / "tables" / "exp1_feature_comparison.csv").exists()


def test_threshold_sensitivity_experiment(tmp_path: Path):
    df = generate_synthetic_dataset(num_games=15, random_seed=42)
    train_df, val_df, test_df = split_by_games(df, 0.70, 0.15, 0.15, random_seed=42)

    fast_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    sens_df = run_threshold_sensitivity_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        thresholds=[150.0, 200.0, 300.0],
        reports_dir=tmp_path / "reports",
        model_params=fast_params,
    )

    assert len(sens_df) == 3
    assert (tmp_path / "reports" / "tables" / "exp2_threshold_sensitivity.csv").exists()
