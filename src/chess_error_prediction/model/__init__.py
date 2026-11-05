"""
Model training, evaluation metrics, and calibration package.
"""

from chess_error_prediction.model.metrics import compute_classification_metrics
from chess_error_prediction.model.calibration import plot_calibration_curve, plot_feature_importance
from chess_error_prediction.model.train import train_xgboost_classifier, save_model

__all__ = [
    "compute_classification_metrics",
    "plot_calibration_curve",
    "plot_feature_importance",
    "train_xgboost_classifier",
    "save_model",
]
