"""
Stage 6 Feature Verification Module (Backend 3).
Audits feature matrices for:
1. XGBoost compatibility (numeric dtypes, no string/object, no NaNs/Infs).
2. Strict Target Leakage prevention (ensuring no post-move variables enter X).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


FORBIDDEN_LEAKAGE_TERMS = [
    "eval_after",
    "eval_loss",
    "is_error",
    "error",
    "blunder",
    "target_move",
    "target_move_uci",
    "move_played",
    "san",
    "uci",
    "fen_after",
]


def audit_target_leakage(feature_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Checks feature names against forbidden post-move leakage patterns.
    """
    violations = []
    for feat in feature_names:
        feat_clean = feat.strip().lower()
        for forbidden in FORBIDDEN_LEAKAGE_TERMS:
            if forbidden == feat_clean or f"_{forbidden}" in feat_clean or f"{forbidden}_" in feat_clean:
                violations.append(f"Feature '{feat}' matches forbidden target leakage term '{forbidden}'!")

    return len(violations) == 0, violations


def audit_features_for_xgboost(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Validates that features in DataFrame are strictly numeric, finite, and well-behaved for XGBoost.
    """
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        return {
            "status": "FAIL",
            "error": f"Missing feature columns in dataframe: {missing_cols}",
        }

    X = df[feature_cols]

    # Dtype check
    non_numeric = [col for col, dt in X.dtypes.items() if not np.issubdtype(dt, np.number)]

    # Null / NaN check
    null_counts = X.isnull().sum().to_dict()
    total_nulls = int(X.isnull().sum().sum())

    # Inf check
    numeric_df = X.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum().to_dict()
    total_infs = int(np.isinf(numeric_df).sum().sum())

    # Target leakage audit
    is_clean, leakage_violations = audit_target_leakage(feature_cols)

    passed = (len(non_numeric) == 0 and total_nulls == 0 and total_infs == 0 and is_clean)

    return {
        "status": "PASS" if passed else "FAIL",
        "num_features_evaluated": len(feature_cols),
        "non_numeric_columns": non_numeric,
        "total_null_entries": total_nulls,
        "total_infinite_entries": total_infs,
        "is_leakage_free": is_clean,
        "leakage_violations": leakage_violations,
        "features_audited": feature_cols,
    }
