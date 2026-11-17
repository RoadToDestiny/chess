"""
Training pipelines for XGBoost (primary) and LogisticRegression (baseline).
Strictly obeys Stage 8 requirements:
- Group-based split respect (Train/Val/Test)
- StandardScaler + OneHotEncoder for LogisticRegression
- Probabilistic calibration (Platt & Isotonic)
- Saving model.json and metadata
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from chess_error_prediction.schemas import ALL_FEATURES
from chess_error_prediction.model.baseline_lr import (
    create_logistic_regression_pipeline,
    get_lr_feature_lists,
)
from chess_error_prediction.model.metrics import compute_classification_metrics
from chess_error_prediction.model.calibration import (
    fit_grouped_calibrators,
    GroupedCalibrationResult,
)


@dataclass
class TrainingResult:
    """Encapsulates trained models, evaluated metrics, and training metadata."""
    xgb_model: xgb.XGBClassifier
    lr_pipeline: Pipeline
    xgb_calibrators: Optional[GroupedCalibrationResult]
    metrics_val: Dict[str, Dict[str, float]]
    metrics_test: Dict[str, Dict[str, float]]
    feature_names: List[str]
    best_iteration: Optional[int]
    params: Dict[str, Any]


def get_default_xgb_params(device: str = "cuda") -> Dict[str, Any]:
    """Returns research default hyperparameters for XGBoost."""
    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if device == "cuda":
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    else:
        params["tree_method"] = "hist"
        params["device"] = "cpu"
    return params


def train_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "is_error",
    group_col: str = "game_id",
    xgb_params: Optional[Dict[str, Any]] = None,
    lr_C: float = 1.0,
    enable_calibration: bool = True,
    device: str = "cuda",
) -> TrainingResult:
    """
    Trains primary XGBoost and baseline LogisticRegression models.
    """
    features = list(feature_cols or ALL_FEATURES)

    # Prepare Train data
    X_train = train_df[features].astype(np.float32)
    y_train = train_df[target_col].astype(int).values
    groups_train = train_df[group_col].values

    # Prepare Val data
    X_val = val_df[features].astype(np.float32)
    y_val = val_df[target_col].astype(int).values

    # Prepare Test data if provided
    X_test = test_df[features].astype(np.float32) if test_df is not None else None
    y_test = test_df[target_col].astype(int).values if test_df is not None else None

    # 1. Setup & Train XGBoost
    params = xgb_params or get_default_xgb_params(device=device)
    xgb_model = xgb.XGBClassifier(**params)

    try:
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    except Exception as e:
        # Fallback to CPU if CUDA fails
        if params.get("device") == "cuda":
            print(f"[WARN] CUDA training failed ({e}). Falling back to CPU.")
            params["device"] = "cpu"
            xgb_model = xgb.XGBClassifier(**params)
            xgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            raise e

    # 2. Setup & Train LogisticRegression Baseline
    lr_pipeline = create_logistic_regression_pipeline(
        all_features=features,
        C=lr_C,
        random_state=42,
    )
    lr_pipeline.fit(X_train, y_train)

    # 3. Probability Calibration on Train via GroupKFold
    xgb_calibrators = None
    if enable_calibration:
        def factory():
            c_params = dict(params)
            # Reduce n_estimators for quick OOF fitting
            c_params["n_estimators"] = min(c_params.get("n_estimators", 100), 100)
            return xgb.XGBClassifier(**c_params)

        try:
            xgb_calibrators = fit_grouped_calibrators(
                X=X_train,
                y=y_train,
                groups=groups_train,
                estimator_factory=factory,
                n_splits=5,
            )
        except Exception as e:
            print(f"[WARN] Grouped calibration fitting skipped: {e}")

    # 4. Predict probabilities on Val
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    lr_val_prob = lr_pipeline.predict_proba(X_val)[:, 1]

    metrics_val = {
        "xgboost": compute_classification_metrics(y_val, xgb_val_prob),
        "logistic_regression": compute_classification_metrics(y_val, lr_val_prob),
    }

    # 5. Predict probabilities on Test if provided
    metrics_test = {}
    if X_test is not None and y_test is not None:
        xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
        lr_test_prob = lr_pipeline.predict_proba(X_test)[:, 1]

        metrics_test["xgboost"] = compute_classification_metrics(y_test, xgb_test_prob)
        metrics_test["logistic_regression"] = compute_classification_metrics(y_test, lr_test_prob)

        if xgb_calibrators is not None:
            platt_test_prob = xgb_calibrators.platt.predict_proba(xgb_test_prob.reshape(-1, 1))[:, 1]
            iso_test_prob = xgb_calibrators.isotonic.transform(xgb_test_prob)
            metrics_test["xgboost_platt"] = compute_classification_metrics(y_test, platt_test_prob)
            metrics_test["xgboost_isotonic"] = compute_classification_metrics(y_test, iso_test_prob)

    best_iter = getattr(xgb_model, "best_iteration", None)

    return TrainingResult(
        xgb_model=xgb_model,
        lr_pipeline=lr_pipeline,
        xgb_calibrators=xgb_calibrators,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        feature_names=features,
        best_iteration=best_iter,
        params=params,
    )


def save_training_artifacts(
    result: TrainingResult,
    models_dir: Path,
    runs_dir: Path,
    model_name: str = "model.json",
    meta_name: str = "model_metadata.json",
    run_name: str = "main_run.json",
) -> Tuple[Path, Path, Path]:
    """
    Saves model and metadata to disk.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_name
    meta_path = models_dir / meta_name
    run_path = runs_dir / run_name

    # Save XGBoost JSON model
    result.xgb_model.save_model(str(model_path))

    # Feature importance dict
    try:
        importance_dict = dict(zip(result.feature_names, [float(x) for x in result.xgb_model.feature_importances_]))
    except Exception:
        importance_dict = {}

    metadata = {
        "model_type": "XGBClassifier",
        "n_features": len(result.feature_names),
        "feature_names": result.feature_names,
        "feature_importances": importance_dict,
        "hyperparameters": {k: str(v) for k, v in result.params.items()},
        "best_iteration": result.best_iteration,
        "metrics_val": result.metrics_val,
        "metrics_test": result.metrics_test,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, meta_path, run_path
