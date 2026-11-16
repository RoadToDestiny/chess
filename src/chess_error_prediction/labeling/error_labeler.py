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
    "played_move",
    "result",
    "termination",
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
    threshold_cp: float = 200.0,
    max_clamped_loss: float = 2500.0,
) -> Dict[str, Any]:
    """
    Integrates raw Stockfish evaluations for before and after move,
    producing eval_loss, is_mate_eval, is_mate_error, and is_error.

    Asymmetric mate handling protocol:
    1. Pre-move forced mate (eval_before is mate): position is already decided,
       marked with is_mate_eval=True and excluded from regression.
    2. Blunder into mate (eval_before normal, eval_after opponent has mate):
       marked as unconditional blunder is_error=1 with loss=max_clamped_loss.
    3. Delivery of mate (eval_before normal, eval_after player has mate):
       sound move with loss=0.0 and is_error=0.
    """
    is_white = color in ("w", "W", 1, "white", "White")
    is_mate_before = bool(eval_before.get("is_mate", False))
    is_mate_after = bool(eval_after.get("is_mate", False))
    score_before = eval_before.get("score_cp")
    score_after = eval_after.get("score_cp")

    if is_mate_before or score_before is None:
        loss = None
        is_error = 0
        is_mate_eval = True
        is_mate_error = False
    elif is_mate_after:
        mate_in = eval_after.get("mate_in")
        # In White POV: positive mate_in means White mates; negative means Black mates
        if mate_in is not None:
            opponent_mates = (mate_in < 0) if is_white else (mate_in > 0)
        elif score_after is not None:
            # Fallback to score_after if mate_in is absent
            opponent_mates = (score_after < 0) if is_white else (score_after > 0)
        else:
            opponent_mates = True

        if opponent_mates:
            loss = float(max_clamped_loss)
            is_error = 1
            is_mate_eval = False
            is_mate_error = True
        else:
            loss = 0.0
            is_error = 0
            is_mate_eval = False
            is_mate_error = False
    elif score_after is None:
        loss = None
        is_error = 0
        is_mate_eval = False
        is_mate_error = False
    else:
        loss = compute_centipawn_loss(float(score_before), float(score_after), color)
        is_error = assign_blunder_label(loss, threshold_cp=threshold_cp)
        is_mate_eval = False
        is_mate_error = False

    return {
        "eval_before": score_before,
        "eval_after": score_after,
        "eval_loss": loss,
        "is_mate_eval": is_mate_eval,
        "is_mate_error": is_mate_error,
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
    mate_errors = int(df.get("is_mate_error", pd.Series(False, index=df.index)).fillna(False).sum())

    return filtered, {
        "total": total,
        "excluded": excluded,
        "saved_mate_errors": mate_errors,
        "excluded_other_mates": excluded,
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
