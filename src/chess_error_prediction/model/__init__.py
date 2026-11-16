"""
chess_error_prediction.model - Model training and evaluation package (Stage 9).
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

__all__ = [
    "create_logistic_regression_pipeline",
    "get_lr_feature_lists",
    "GroupedCalibrationResult",
    "evaluate_calibrators",
    "fit_grouped_calibrators",
]
