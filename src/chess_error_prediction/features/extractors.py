"""
Feature extraction from chess positions and game metadata.
Extracts structural, tactical, positional, and player/clock features known BEFORE the move.
"""

from typing import Any, Dict, Optional
import chess


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def extract_board_features(board: chess.Board) -> Dict[str, Any]:
    """
    Extracts structural and tactical features from a chess.Board state BEFORE a move is executed.
    All evaluations and balances are oriented towards the player whose turn it is.
    """
    active_color = board.turn
    opponent_color = not active_color

    # Count pawns and non-king pieces
    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    pawn_count = white_pawns + black_pawns

    white_pieces = sum(len(board.pieces(pt, chess.WHITE)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    black_pieces = sum(len(board.pieces(pt, chess.BLACK)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    piece_count = white_pieces + black_pieces

    # Material balance from active player's perspective
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in PIECE_VALUES.items())
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in PIECE_VALUES.items())
    if active_color == chess.WHITE:
        material_balance = float(white_material - black_material)
    else:
        material_balance = float(black_material - white_material)

    # Tactical indicators: legal checks and legal captures
    legal_checks_count = 0
    legal_captures_count = 0
    for move in board.legal_moves:
        if board.gives_check(move):
            legal_checks_count += 1
        if board.is_capture(move):
            legal_captures_count += 1

    # Attacked friendly pieces count
    attacked_pieces_count = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == active_color:
            if board.is_attacked_by(opponent_color, sq):
                attacked_pieces_count += 1

    # Game phase based on full move number (1-10: opening, 11-30: middlegame, 31+: endgame)
    fullmove = board.fullmove_number
    if fullmove <= 10:
        game_phase = 1  # Opening
    elif fullmove <= 30:
        game_phase = 2  # Middlegame
    else:
        game_phase = 3  # Endgame

    return {
        "material_balance": material_balance,
        "piece_count": piece_count,
        "pawn_count": pawn_count,
        "legal_checks_count": legal_checks_count,
        "legal_captures_count": legal_captures_count,
        "attacked_pieces_count": attacked_pieces_count,
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
