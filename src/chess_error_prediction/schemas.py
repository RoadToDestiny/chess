"""
Single source of truth for dataset schema, column definitions, and data contracts.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os


METADATA_COLUMNS: List[str] = [
    "game_id",
    "position_id",
    "ply",
    "move_number",
    "color",
]

POSITION_FEATURES: List[str] = [
    "eval_before",
    "eval_diff_prev_move",
    "has_prev_move",
    "material_balance",
    "piece_count",
    "pawn_count",
    "legal_checks_count",
    "legal_captures_count",
    "attacked_pieces_count",
    "move_number",
    "game_phase",
]

META_FEATURES: List[str] = [
    "player_rating",
    "opponent_rating",
    "rating_diff",
    "time_control_base",
    "time_control_inc",
]

TARGET_COLUMNS: List[str] = [
    "eval_after",
    "eval_loss",
    "is_error",
]

OPTIONAL_COLUMNS: List[str] = [
    "fen",
    "clock_remaining",
    "time_spent_prev_move",
    "is_mate_eval",
    "is_mate_error",
]

ALL_FEATURES: List[str] = POSITION_FEATURES + META_FEATURES
ALL_REQUIRED_COLUMNS: List[str] = METADATA_COLUMNS + POSITION_FEATURES + META_FEATURES + TARGET_COLUMNS


@dataclass
class PositionRecord:
    """Represents a single position observation in the blunder dataset."""
    game_id: str
    ply: int
    move_number: int
    color: str  # 'w' or 'b'

    # Features (known BEFORE move)
    eval_before: float
    eval_diff_prev_move: float
    material_balance: float
    piece_count: int
    pawn_count: int
    legal_checks_count: int
    legal_captures_count: int
    attacked_pieces_count: int
    has_prev_move: int
    game_phase: int

    # Player & Time Features
    player_rating: float
    opponent_rating: float
    rating_diff: float
    time_control_base: float
    time_control_inc: float

    # Target & Post-move Evaluation
    eval_after: Optional[float] = None
    eval_loss: Optional[float] = None
    is_error: int = 0
    is_mate_eval: bool = False
    is_mate_error: bool = False
    fen: Optional[str] = None
    clock_remaining: Optional[float] = None
    time_spent_prev_move: Optional[float] = None

    @property
    def position_id(self) -> str:
        """Composite unique identifier for the position within the research dataset."""
        return f"{self.game_id}_{self.ply}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "position_id": self.position_id,
            "ply": self.ply,
            "move_number": self.move_number,
            "color": self.color,
            "eval_before": self.eval_before,
            "eval_diff_prev_move": self.eval_diff_prev_move,
            "has_prev_move": self.has_prev_move,
            "material_balance": self.material_balance,
            "piece_count": self.piece_count,
            "pawn_count": self.pawn_count,
            "legal_checks_count": self.legal_checks_count,
            "legal_captures_count": self.legal_captures_count,
            "attacked_pieces_count": self.attacked_pieces_count,
            "game_phase": self.game_phase,
            "player_rating": self.player_rating,
            "opponent_rating": self.opponent_rating,
            "rating_diff": self.rating_diff,
            "time_control_base": self.time_control_base,
            "time_control_inc": self.time_control_inc,
            "eval_after": self.eval_after,
            "eval_loss": self.eval_loss,
            "is_error": self.is_error,
            "is_mate_eval": self.is_mate_eval,
            "is_mate_error": self.is_mate_error,
            "fen": self.fen,
            "clock_remaining": self.clock_remaining,
            "time_spent_prev_move": self.time_spent_prev_move,
        }
