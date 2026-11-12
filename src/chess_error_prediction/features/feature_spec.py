"""
Feature specification and metadata for chess error prediction.
Defines feature subsets, types, defaults, and descriptions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """Specification of features used in modeling and experiments."""
    position_features: List[str] = field(default_factory=lambda: [
        "eval_before",
        "eval_diff_prev_move",
        "material_balance",
        "piece_count",
        "pawn_count",
        "legal_checks_count",
        "legal_captures_count",
        "attacked_pieces_count",
        "game_phase",
    ])
    meta_features: List[str] = field(default_factory=lambda: [
        "player_rating",
        "opponent_rating",
        "rating_diff",
        "time_control_base",
        "time_control_inc",
    ])
    optional_features: List[str] = field(default_factory=lambda: [
        "clock_remaining",
    ])

    @property
    def all_features(self) -> List[str]:
        return self.position_features + self.meta_features


def get_default_feature_spec() -> FeatureSpec:
    return FeatureSpec()


FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "eval_before": "Stockfish evaluation of current position before move (centipawns from active player POV).",
    "eval_diff_prev_move": "Change in Stockfish evaluation on the preceding half-move.",
    "material_balance": "Difference in total piece values (P=1, N/B=3, R=5, Q=9) from active player POV.",
    "piece_count": "Total non-king pieces on the board.",
    "pawn_count": "Total pawns on the board (0..16).",
    "legal_checks_count": "Number of legal moves giving check to the opponent.",
    "legal_captures_count": "Number of legal capture moves available.",
    "attacked_pieces_count": "Number of friendly pieces attacked by enemy pieces.",
    "game_phase": "Phase of game (1: opening moves 1-10, 2: middlegame moves 11-30, 3: endgame moves 31+).",
    "player_rating": "Elo rating of active player making move.",
    "opponent_rating": "Elo rating of opponent.",
    "rating_diff": "Rating difference (player_rating - opponent_rating).",
    "time_control_base": "Base time in seconds (e.g., 600s for 10 min).",
    "time_control_inc": "Time increment per move in seconds.",
    "clock_remaining": "Remaining clock time for active player in seconds (optional).",
}

DEFAULT_IMPUTATIONS: Dict[str, float] = {
    "eval_before": 0.0,
    "eval_diff_prev_move": 0.0,
    "material_balance": 0.0,
    "piece_count": 14.0,
    "pawn_count": 8.0,
    "legal_checks_count": 0.0,
    "legal_captures_count": 0.0,
    "attacked_pieces_count": 0.0,
    "game_phase": 2.0,
    "player_rating": 1500.0,
    "opponent_rating": 1500.0,
    "rating_diff": 0.0,
    "time_control_base": 600.0,
    "time_control_inc": 0.0,
    "clock_remaining": 300.0,
}


def impute_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in feature columns according to the research policy."""
    df_clean = df.copy()
    for col, default_val in DEFAULT_IMPUTATIONS.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(default_val)
    return df_clean
