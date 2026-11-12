"""
Feature extraction from chess positions and game metadata.
Extracts structural, tactical, positional, and player/clock features known BEFORE the move.
Compatible with Backend 2's specifications and the unified research schema.
"""

from typing import Any, Dict, Optional
import chess

from chess_error_prediction.features.feature_spec import PIECE_VALUES


def extract_board_features(board: chess.Board) -> Dict[str, Any]:
    """
    Extracts structural and tactical features from chess.Board state BEFORE move execution.
    All evaluations and balances are oriented towards the player whose turn it is.
    """
    active_color = board.turn
    opponent_color = not active_color

    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    pawn_count = white_pawns + black_pawns

    white_pieces = sum(len(board.pieces(pt, chess.WHITE)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    black_pieces = sum(len(board.pieces(pt, chess.BLACK)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    piece_count = white_pieces + black_pieces

    # Material balance from active player POV
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
    material_balance = float(white_material - black_material) if active_color == chess.WHITE else float(black_material - white_material)

    legal_checks_count = sum(1 for move in board.legal_moves if board.gives_check(move))
    legal_captures_count = sum(1 for move in board.legal_moves if board.is_capture(move))

    # Attacked friendly pieces (excluding king)
    attacked_pieces_count = 0
    attacked_pieces_value = 0
    for sq, piece in board.piece_map().items():
        if piece.color == active_color and piece.piece_type != chess.KING:
            if board.is_attacked_by(opponent_color, sq):
                attacked_pieces_count += 1
                attacked_pieces_value += piece_values.get(piece.piece_type, 0)

    # Game phase (1: opening moves <=10, 2: middlegame moves 11-30, 3: endgame moves 31+)
    fullmove = board.fullmove_number
    if fullmove <= 10:
        game_phase = 1
    elif fullmove <= 30:
        game_phase = 2
    else:
        game_phase = 3

    return {
        "material_balance": material_balance,
        "piece_count": piece_count,
        "pawn_count": pawn_count,
        "legal_checks_count": legal_checks_count,
        "legal_captures_count": legal_captures_count,
        "attacked_pieces_count": attacked_pieces_count,
        "attacked_pieces_value": attacked_pieces_value,
        "game_phase": game_phase,
    }


def extract_all_features(
    board: chess.Board,
    eval_before_player_pov: float,
    eval_diff_prev_move: float,
    white_elo: float,
    black_elo: float,
    time_control_base: float = 600.0,
    time_control_inc: float = 0.0,
    clock_remaining: Optional[float] = None
) -> Dict[str, Any]:
    """
    Combines board features with evaluation and metadata features into a unified dictionary.
    Guarantees no post-move information is included.
    """
    features = extract_board_features(board)

    is_white = (board.turn == chess.WHITE)
    player_rating = float(white_elo if is_white else black_elo)
    opponent_rating = float(black_elo if is_white else white_elo)
    rating_diff = player_rating - opponent_rating

    features.update({
        "eval_before": float(eval_before_player_pov),
        "eval_diff_prev_move": float(eval_diff_prev_move),
        "player_rating": player_rating,
        "opponent_rating": opponent_rating,
        "rating_diff": rating_diff,
        "time_control_base": float(time_control_base),
        "time_control_inc": float(time_control_inc),
        "clock_remaining": clock_remaining,
    })
    return features
