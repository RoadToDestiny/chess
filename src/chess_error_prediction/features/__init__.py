"""
Feature extraction and specification package.
"""

from chess_error_prediction.features.feature_spec import (
    FeatureSpec,
    get_default_feature_spec,
    FEATURE_DESCRIPTIONS,
    impute_missing_features,
)
from chess_error_prediction.features.extractors import (
    extract_board_features,
    extract_all_features,
)

__all__ = [
    "FeatureSpec",
    "get_default_feature_spec",
    "FEATURE_DESCRIPTIONS",
    "impute_missing_features",
    "extract_board_features",
    "extract_all_features",
]
