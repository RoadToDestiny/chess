"""
Error labeling logic based on Stockfish centipawn loss and configurable thresholds.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


FORBIDDEN_LEAKAGE_SUBSTRINGS = [
    "eval_after",
    "eval_loss",
    "is_error",
    "blunder",
    "target",
    "label",
]


def compute_centipawn_loss(
    eval_before_white: float,
    eval_after_white: float,
    color: Union[str, int]
) -> float:
    """
    Computes centipawn loss from the perspective of the player making the move,
    given evaluations strictly from White's POV.

    - For White: eval_loss = eval_before_white - eval_after_white
    - For Black: eval_loss = eval_after_white - eval_before_white
    """
    if color not in ("w", "W", 1, "white", "White", "b", "B", 0, "black", "Black"):
        raise ValueError(f"Unsupported player color: {color!r}")
    is_white = color in ("w", "W", 1, "white", "White")
    if not math.isfinite(eval_before_white) or not math.isfinite(eval_after_white):
        raise ValueError("Position evaluations must be finite numbers.")
    if is_white:
        loss = eval_before_white - eval_after_white
    else:
        loss = eval_after_white - eval_before_white

    return float(loss)


def assign_blunder_label(
    eval_loss: Optional[float],
    threshold_cp: float = 200.0
) -> int:
    """
    Returns 1 if centipawn loss is >= threshold_cp, otherwise 0.
    """
    if threshold_cp < 0:
        raise ValueError("Error threshold must be non-negative.")
    if eval_loss is None or pd.isna(eval_loss):
        return 0
    if not math.isfinite(float(eval_loss)):
        return 0
    return 1 if eval_loss >= threshold_cp else 0


def label_position_evaluations(
    eval_before: Dict[str, Any],
    eval_after: Dict[str, Any],
    color: Union[str, int],
    threshold_cp: float = 200.0
) -> Dict[str, Any]:
    """
    Integrates raw Stockfish evaluations for before and after move,
    producing eval_loss, is_mate_eval, and is_error.
    """
    is_mate_eval = bool(eval_before.get("is_mate", False) or eval_after.get("is_mate", False))
    score_before = eval_before.get("score_cp")
    score_after = eval_after.get("score_cp")

    if is_mate_eval or score_before is None or score_after is None:
        loss = None
        is_error = 0
    else:
        loss = compute_centipawn_loss(float(score_before), float(score_after), color)
        is_error = assign_blunder_label(loss, threshold_cp=threshold_cp)

    return {
        "eval_before": score_before,
        "eval_after": score_after,
        "eval_loss": loss,
        "is_mate_eval": is_mate_eval,
        "is_error": is_error,
    }


def filter_mate_records(
    df: pd.DataFrame,
    mate_col: str = "is_mate_eval"
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """
    Filters out positions marked with mate evaluations.
    """
    total = len(df)
    if mate_col not in df.columns:
        return df.copy(), {"total": total, "excluded": 0, "pct_excluded": 0.0}

    filtered = df[~df[mate_col].fillna(False)].copy()
    excluded = total - len(filtered)
    pct = (excluded / total * 100.0) if total > 0 else 0.0

    return filtered, {
        "total": total,
        "excluded": excluded,
        "pct_excluded": round(pct, 2),
    }


def audit_features_for_leakage(feature_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Verifies that no target or post-move variables are present in feature lists.
    """
    violations: List[str] = []
    for col in feature_names:
        col_lower = col.lower()
        for forbidden in FORBIDDEN_LEAKAGE_SUBSTRINGS:
            if forbidden in col_lower:
                violations.append(f"Feature '{col}' contains forbidden substring '{forbidden}' (target leakage).")
    return len(violations) == 0, violations
