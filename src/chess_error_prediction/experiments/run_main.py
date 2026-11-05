"""
Main research experiment: Trains full XGBoost model and evaluates on held-out test split.
Tests Hypothesis H4 (ROC-AUC >= 0.75).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import pandas as pd

from chess_error_prediction.model.train import train_xgboost_classifier, save_model
from chess_error_prediction.model.metrics import compute_classification_metrics
from chess_error_prediction.model.calibration import plot_calibration_curve, plot_feature_importance
from chess_error_prediction.schemas import ALL_FEATURES


def run_main_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts_dir: Path = Path("artifacts"),
    reports_dir: Path = Path("reports"),
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Executes the primary research experiment:
    1. Trains XGBoost on train_df using validation on val_df
    2. Evaluates probabilities on test_df
    3. Exports model.json, calibration curves, feature importance, and test metrics
    """
    feature_cols = ALL_FEATURES

    # 1. Train model
    model, val_metrics = train_xgboost_classifier(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        params=model_params,
    )

    # 2. Evaluate on test split
    X_test = test_df[feature_cols]
    y_test = test_df["is_error"].to_numpy()
    y_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = compute_classification_metrics(y_test, y_prob)

    # 3. Ensure directories exist
    models_dir = artifacts_dir / "models"
    runs_dir = artifacts_dir / "runs"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"

    for d in [models_dir, runs_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 4. Save model artifact
    save_model(model, models_dir, feature_cols, model_params or {})

    # 5. Save metrics table
    res_df = pd.DataFrame([{
        "experiment": "main_xgboost_model",
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "test_error_ratio": round(float(y_test.mean()), 4),
        **test_metrics,
    }])
    res_df.to_csv(tables_dir / "main_experiment_metrics.csv", index=False)

    # 6. Save run metadata
    with open(runs_dir / "main_run.json", "w", encoding="utf-8") as f:
        json.dump({
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "features": feature_cols,
            "model_params": model_params,
        }, f, indent=2)

    # 7. Generate diagnostic plots
    plot_calibration_curve(
        y_test,
        y_prob,
        save_path=figures_dir / "calibration_curve.png",
        model_name="XGBoost Full Model",
    )

    if hasattr(model, "feature_importances_"):
        df_imp = plot_feature_importance(
            feature_names=feature_cols,
            importances=model.feature_importances_,
            save_path=figures_dir / "feature_importance.png",
        )
        df_imp.to_csv(tables_dir / "feature_importance_ranking.csv", index=False)

    return test_metrics
