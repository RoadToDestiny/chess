"""
LogisticRegression baseline pipeline with strict preprocessing:
- OneHotEncoding for categorical features (game_phase)
- StandardScaler for numeric features
- Exclusion of collinear rating_diff feature
- L2-regularization (penalty='l2')
- Zero-leakage fit strictly on Train split
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from chess_error_prediction.schemas import ALL_FEATURES


def get_lr_feature_lists(
    all_features: Optional[List[str]] = None,
) -> tuple[List[str], List[str], List[str]]:
    """
    Partitions feature space for LogisticRegression:
    - categorical_cols: ['game_phase']
    - dropped_cols: ['rating_diff'] (exact collinearity with player_rating and opponent_rating)
    - numeric_cols: remaining features to be scaled with StandardScaler
    """
    features = list(all_features or ALL_FEATURES)
    dropped_cols = [c for c in ["rating_diff"] if c in features]
    categorical_cols = [c for c in ["game_phase"] if c in features]
    numeric_cols = [
        c for c in features if c not in dropped_cols and c not in categorical_cols
    ]
    return numeric_cols, categorical_cols, dropped_cols


def create_logistic_regression_pipeline(
    all_features: Optional[List[str]] = None,
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Creates an end-to-end scikit-learn Pipeline with:
    1. StandardScaler for numeric features
    2. OneHotEncoder for game_phase
    3. Dropping rating_diff to eliminate strict multicollinearity
    4. L2-regularized LogisticRegression classifier
    """
    numeric_cols, categorical_cols, _ = get_lr_feature_lists(all_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline
