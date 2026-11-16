"""
Stage 5 Verification Module (Backend 3).
Verifies binary target variable correctness and calculates class imbalance distributions.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def verify_binary_target_integrity(y: pd.Series) -> Dict[str, Any]:
    """
    Asserts target is strictly binary {0, 1} with no nulls, floats, or unexpected labels.
    """
    total = len(y)
    null_count = int(y.isnull().sum())
    unique_vals = set(y.dropna().unique())

    is_binary = unique_vals.issubset({0, 1})
    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())

    pos_ratio = pos_count / max(1, total)
    neg_ratio = neg_count / max(1, total)

    # Imbalance scale_pos_weight = negative_examples / positive_examples
    scale_pos_weight = float(neg_count / max(1, pos_count))

    status = "PASS" if (is_binary and null_count == 0) else "FAIL"

    return {
        "status": status,
        "total_examples": total,
        "null_count": null_count,
        "unique_values": list(unique_vals),
        "is_strictly_binary": is_binary,
        "positive_count_blunders": pos_count,
        "negative_count_non_blunders": neg_count,
        "blunder_percentage": round(pos_ratio * 100.0, 2),
        "recommended_scale_pos_weight": round(scale_pos_weight, 2),
    }


def analyze_threshold_sensitivity_distribution(
    eval_losses: pd.Series,
    thresholds: List[float] = [150.0, 200.0, 300.0]
) -> pd.DataFrame:
    """
    Computes class distributions across various blunder definition thresholds.
    """
    records = []
    total = len(eval_losses)

    for thresh in thresholds:
        pos = int((eval_losses >= thresh).sum())
        neg = total - pos
        ratio = pos / max(1, total)
        spw = neg / max(1, pos)

        records.append({
            "threshold_cp": thresh,
            "blunder_count": pos,
            "non_blunder_count": neg,
            "blunder_rate_pct": round(ratio * 100.0, 2),
            "scale_pos_weight": round(spw, 2),
        })

    return pd.DataFrame(records)
