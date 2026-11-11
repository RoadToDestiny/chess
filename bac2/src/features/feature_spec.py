
"""
Specification of chess features.

All features must be calculated from the position
before the target move.

Stockfish evaluations are provisionally represented
from White's point of view:
positive = advantage for White,
negative = advantage for Black.
"""

FEATURE_NAMES = [
    "stockfish_eval_cp",
    "material_balance",
    "white_piece_count",
    "black_piece_count",
    "white_pawn_count",
    "black_pawn_count",
    "legal_check_count",
    "legal_capture_count",
    "attacked_piece_count",
    "attacked_piece_value",
    "move_number",
    "game_phase",
    "previous_eval_change_cp",
    "white_elo",
    "black_elo",
    "elo_difference",
    "time_control_base",
    "time_control_increment",
    "player_time_remaining",
]

PIECE_VALUES = {
    "pawn": 1,
    "knight": 3,
    "bishop": 3,
    "rook": 5,
    "queen": 9,
    "king": 0,
}

GAME_PHASES = {
    "opening": 0,
    "middlegame": 1,
    "endgame": 2,
}

TARGET_COLUMN = "error"

IDENTIFIER_COLUMNS = [
    "game_id",
    "position_id",
    "ply",
    "fen",
    "target_move_uci",
]

# Identifiers are not model features.
# The target column is also excluded from X.
