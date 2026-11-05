"""
Model training pipeline for XGBoost with class balance management and serialization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pandas as pd

from chess_error_prediction.model.metrics import compute_classification_metrics


def train_xgboost_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "is_error",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, float]]:
    """
    Trains an XGBoost classifier with class balancing and validation monitoring.
    """
    import xgboost as xgb

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # Calculate class imbalance to compensate for error rarity (~5-10%)
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    default_spw = float(neg_count / max(1, pos_count))

    default_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": default_spw,
        "eval_metric": "auc",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_classification_metrics(y_val.to_numpy(), val_probs)

    return model, val_metrics


def save_model(
    model: Any,
    save_dir: Path,
    feature_names: List[str],
    hyperparams: Dict[str, Any]
) -> Path:
    """
    Saves trained XGBoost model in JSON format along with feature list and hyperparameters metadata.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "model.json"
    model.save_model(str(model_path))

    metadata = {
        "features": feature_names,
        "hyperparameters": hyperparams,
    }
    meta_path = save_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return model_path
