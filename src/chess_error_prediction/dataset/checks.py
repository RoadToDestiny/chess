"""
Dataset checks, schema validation, and leakage detection.
Integrates Backend 2's validation rules and Backend 3's target leakage checks.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import chess


FORBIDDEN_LEAKAGE_SUBSTRINGS = [
    "eval_after",
    "eval_loss",
    "is_error",
    "error",
    "blunder",
    "target_move",
    "target_move_uci",
]


def check_for_target_leakage(feature_cols: List[str]) -> Tuple[bool, List[str]]:
    """
    Verifies that no target, post-move, or evaluation-loss variables leaked into the feature space.
    """
    violations = []
    for col in feature_cols:
        col_lower = col.lower().strip()
        for forbidden in FORBIDDEN_LEAKAGE_SUBSTRINGS:
            if forbidden == col_lower or f"_{forbidden}" in col_lower or f"{forbidden}_" in col_lower:
                violations.append(f"Feature '{col}' contains forbidden pattern '{forbidden}' (target leakage).")
    return len(violations) == 0, violations


def validate_labeled_record(record: Dict[str, Any]) -> bool:
    """
    Validates single labeled record (from Backend 2 checks):
    Verifies position_id, player_color, FEN validity, legal moves, and error binary flag.
    """
    required_fields = {"game_id", "position_id", "ply", "player_color", "fen", "target_move_uci", "error"}
    missing = required_fields - record.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if record["error"] not in (0, 1):
        raise ValueError("Field 'error' must be 0 or 1")

    if record["player_color"] not in ("white", "black", "w", "b"):
        raise ValueError("Invalid player_color")

    board = chess.Board(record["fen"])
    try:
        move = chess.Move.from_uci(record["target_move_uci"])
    except ValueError as exc:
        raise ValueError("Invalid target_move_uci") from exc

    if move not in board.legal_moves:
        raise ValueError("Move is illegal in position")

    return True
