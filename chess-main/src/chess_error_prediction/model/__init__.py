"""
chess_error_prediction.model - Model training and evaluation package.
"""

from chess_error_prediction.model.baseline_lr import (
    create_logistic_regression_pipeline,
    get_lr_feature_lists,
)
from chess_error_prediction.model.calibration import (
    GroupedCalibrationResult,
    evaluate_calibrators,
    fit_grouped_calibrators,
)
from chess_error_prediction.model.metrics import compute_classification_metrics
from chess_error_prediction.model.train import (
    TrainingResult,
    get_default_xgb_params,
    train_models,
    save_training_artifacts,
)

__all__ = [
    "create_logistic_regression_pipeline",
    "get_lr_feature_lists",
    "GroupedCalibrationResult",
    "evaluate_calibrators",
    "fit_grouped_calibrators",
    "compute_classification_metrics",
    "TrainingResult",
    "get_default_xgb_params",
    "train_models",
    "save_training_artifacts",
]
