"""
Experiment 2: Threshold Sensitivity Analysis (150, 200, 300 centipawns).
Tests Hypothesis H3 without requiring repeated Stockfish engine calls.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from chess_error_prediction.model.train import train_xgboost_classifier
from chess_error_prediction.model.metrics import compute_classification_metrics
from chess_error_prediction.schemas import ALL_FEATURES


def run_threshold_sensitivity_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    reports_dir: Path = Path("reports"),
    model_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Re-labels target variable based on different centipawn loss thresholds (150, 200, 300 cp)
    and evaluates model sensitivity in ROC-AUC, PR-AUC, and Brier Score.
    """
    thresholds = thresholds or [150.0, 200.0, 300.0]
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = []
    feature_cols = ALL_FEATURES

    for thresh in thresholds:
        # Re-derive targets on data copies without recalculating engine positions
        tr = train_df.copy()
        vl = val_df.copy()
        ts = test_df.copy()

        tr["is_error"] = (tr["eval_loss"] >= thresh).astype(int)
        vl["is_error"] = (vl["eval_loss"] >= thresh).astype(int)
        ts["is_error"] = (ts["eval_loss"] >= thresh).astype(int)

        error_rate_test = float(ts["is_error"].mean())
        error_count_test = int(ts["is_error"].sum())

        model, val_metrics = train_xgboost_classifier(
            train_df=tr,
            val_df=vl,
            feature_cols=feature_cols,
            params=model_params,
        )

        probs = model.predict_proba(ts[feature_cols])[:, 1]
        metrics = compute_classification_metrics(ts["is_error"].to_numpy(), probs)

        results.append({
            "threshold_cp": thresh,
            "test_positive_blunders": error_count_test,
            "test_error_rate": round(error_rate_test, 4),
            "val_roc_auc": val_metrics["roc_auc"],
            **metrics,
        })

    sens_df = pd.DataFrame(results)
    sens_df.to_csv(tables_dir / "exp2_threshold_sensitivity.csv", index=False)
    return sens_df
