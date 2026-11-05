"""
Experiment 1: Feature Ablation Study.
Compares a position-only model against a full model with player ratings and time control.
Tests Hypothesis H2.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from chess_error_prediction.model.train import train_xgboost_classifier
from chess_error_prediction.model.metrics import compute_classification_metrics
from chess_error_prediction.schemas import POSITION_FEATURES, ALL_FEATURES


def run_feature_sets_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reports_dir: Path = Path("reports"),
    model_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Trains and compares two models:
    - Model 1 (Baseline): Positional features only (material, checks, attacks, eval)
    - Model 2 (Full): Positional features + Player ratings + Time controls
    """
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    y_test = test_df["is_error"].to_numpy()
    results = []

    # Model 1: Positional only
    model_pos, val_metrics_pos = train_xgboost_classifier(
        train_df=train_df,
        val_df=val_df,
        feature_cols=POSITION_FEATURES,
        params=model_params,
    )
    probs_pos = model_pos.predict_proba(test_df[POSITION_FEATURES])[:, 1]
    metrics_pos = compute_classification_metrics(y_test, probs_pos)
    results.append({
        "feature_set": "Positional Only",
        "num_features": len(POSITION_FEATURES),
        "val_roc_auc": val_metrics_pos["roc_auc"],
        **metrics_pos,
    })

    # Model 2: Full (Positional + Meta)
    model_full, val_metrics_full = train_xgboost_classifier(
        train_df=train_df,
        val_df=val_df,
        feature_cols=ALL_FEATURES,
        params=model_params,
    )
    probs_full = model_full.predict_proba(test_df[ALL_FEATURES])[:, 1]
    metrics_full = compute_classification_metrics(y_test, probs_full)
    results.append({
        "feature_set": "Positional + Ratings + Time (Full)",
        "num_features": len(ALL_FEATURES),
        "val_roc_auc": val_metrics_full["roc_auc"],
        **metrics_full,
    })

    comp_df = pd.DataFrame(results)

    # Compute delta improvements
    if len(comp_df) == 2:
        delta_roc = comp_df.loc[1, "roc_auc"] - comp_df.loc[0, "roc_auc"]
        delta_pr = comp_df.loc[1, "pr_auc"] - comp_df.loc[0, "pr_auc"]
        comp_df["delta_roc_auc_vs_baseline"] = [0.0, round(delta_roc, 4)]
        comp_df["delta_pr_auc_vs_baseline"] = [0.0, round(delta_pr, 4)]

    comp_df.to_csv(tables_dir / "exp1_feature_comparison.csv", index=False)
    return comp_df
