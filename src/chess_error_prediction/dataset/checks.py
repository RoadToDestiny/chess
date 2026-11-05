"""
Dataset validation, schema checks, and target leakage detection.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

from chess_error_prediction.schemas import (
    ALL_REQUIRED_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
)

FORBIDDEN_LEAKAGE_SUBSTRINGS = [
    "eval_after",
    "eval_loss",
    "is_error",
    "blunder",
    "target",
    "label",
]


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that a DataFrame conforms to the research schema requirements.
    """
    errors: List[str] = []

    # 1. Check required column presence
    missing = [c for c in ALL_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return False, errors

    # 2. Check null values in critical fields
    critical = ["game_id", "ply", "is_error", "eval_before", "eval_loss"]
    for col in critical:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null/NaN values.")

    # 3. Check target values
    unique_targets = set(df["is_error"].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        errors.append(f"Target column 'is_error' contains invalid values: {unique_targets}. Must be 0 or 1.")

    # 4. Check logical ranges
    if (df["pawn_count"] < 0).any() or (df["pawn_count"] > 16).any():
        errors.append("Column 'pawn_count' has values outside [0, 16].")

    if (df["piece_count"] < 0).any() or (df["piece_count"] > 32).any():
        errors.append("Column 'piece_count' has values outside [0, 32].")

    if (df["ply"] < 1).any():
        errors.append("Column 'ply' must contain positive integers >= 1.")

    # 5. Check for duplicate positions in the same game
    duplicates = df.duplicated(subset=["game_id", "ply"]).sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate (game_id, ply) rows.")

    return len(errors) == 0, errors


def check_for_target_leakage(feature_cols: List[str]) -> Tuple[bool, List[str]]:
    """
    Verifies that no target, post-move, or evaluation-loss variables leaked into the feature space.
    """
    violations: List[str] = []
    for col in feature_cols:
        col_lower = col.lower()
        for forbidden in FORBIDDEN_LEAKAGE_SUBSTRINGS:
            if forbidden in col_lower:
                violations.append(f"Feature '{col}' contains forbidden pattern '{forbidden}' (potential target leakage).")
    return len(violations) == 0, violations


def compute_dataset_summary(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Computes key descriptive statistics of the dataset for logging and paper tables.
    """
    total_positions = len(df)
    total_games = int(df["game_id"].nunique()) if "game_id" in df.columns else 0
    blunder_count = int((df["is_error"] == 1).sum()) if "is_error" in df.columns else 0
    blunder_rate = (blunder_count / total_positions) if total_positions > 0 else 0.0

    return {
        "total_games": total_games,
        "total_positions": total_positions,
        "blunder_count": blunder_count,
        "non_blunder_count": total_positions - blunder_count,
        "blunder_rate": round(float(blunder_rate), 4),
        "mean_positions_per_game": round(total_positions / max(1, total_games), 1),
    }
