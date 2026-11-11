
"""
Chess feature extraction.

All features are calculated from the position
before the target move.

Stockfish evaluations must use White's point of view:
positive = advantage for White,
negative = advantage for Black.
"""

import re

import chess

from src.features.feature_spec import PIECE_VALUES


def parse_rating(value):
    """Convert player rating to integer or None."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_time_control(value):
    """
    Parse simple PGN time controls:
    600+5 -> base=600, increment=5
    600   -> base=600, increment=0

    Complex formats return None values.
    """
    if not value:
        return None, None

    match = re.fullmatch(r"(\d+)(?:\+(\d+))?", str(value))

    if not match:
        return None, None

    base = int(match.group(1))
    increment = int(match.group(2) or 0)

    return base, increment



def calculate_material_balance(board):
    """
    White material minus Black material.

    Pawn = 1
    Knight = 3
    Bishop = 3
    Rook = 5
    Queen = 9
    King = 0
    """

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    balance = 0

    for piece_type, value in piece_values.items():
        white_count = len(
            board.pieces(piece_type, chess.WHITE)
        )

        black_count = len(
            board.pieces(piece_type, chess.BLACK)
        )

        balance += (white_count - black_count) * value

    return balance



def count_pieces(board, color, piece_type):
    """Count pieces of a specified type and color."""
    return len(board.pieces(piece_type, color))


def count_legal_checks(board):
    """Count legal moves that give check."""
    return sum(
        1 for move in board.legal_moves
        if board.gives_check(move)
    )


def count_legal_captures(board):
    """Count legal capture moves."""
    return sum(
        1 for move in board.legal_moves
        if board.is_capture(move)
    )


def calculate_attacked_pieces(board):
    """
    Count attacked non-king pieces belonging to
    the player whose turn it is.

    Returns:
        (number of attacked pieces, total material value)
    """

    player_color = board.turn
    opponent_color = not player_color

    attacked_count = 0
    attacked_value = 0

    for square, piece in board.piece_map().items():

        if piece.color != player_color:
            continue

        if piece.piece_type == chess.KING:
            continue

        if board.is_attacked_by(opponent_color, square):
            attacked_count += 1
            attacked_value += PIECE_VALUES[
                chess.piece_name(piece.piece_type)
            ]

    return attacked_count, attacked_value


def determine_game_phase(board):
    """
    Determine game phase by remaining non-pawn material.

    Opening: >= 40
    Middlegame: 17-39
    Endgame: <= 16

    Returns:
        0 = opening
        1 = middlegame
        2 = endgame
    """

    non_pawn_types = (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    )

    material = 0

    for piece_type in non_pawn_types:
        value = PIECE_VALUES[
            chess.piece_name(piece_type)
        ]

        material += len(
            board.pieces(piece_type, chess.WHITE)
        ) * value

        material += len(
            board.pieces(piece_type, chess.BLACK)
        ) * value

    if material >= 40:
        return 0

    if material > 16:
        return 1

    return 2


def extract_features(
    board,
    metadata,
    stockfish_eval_cp=None,
    previous_eval_cp=None,
    player_time_remaining=None,
):
    """
    Extract model features from the current board position.

    Args:
        board: Position before the target move.
        metadata: PGN headers or a dictionary with game metadata.
        stockfish_eval_cp: Current Stockfish score from White's POV.
        previous_eval_cp: Previous position score from White's POV.
        player_time_remaining: Remaining time in seconds, if available.

    Returns:
        Dictionary containing model features.
    """

    player_color = board.turn

    white_elo = parse_rating(metadata.get("WhiteElo"))
    black_elo = parse_rating(metadata.get("BlackElo"))

    if player_color == chess.WHITE:
        player_elo = white_elo
        opponent_elo = black_elo
    else:
        player_elo = black_elo
        opponent_elo = white_elo

    if player_elo is not None and opponent_elo is not None:
        elo_difference = player_elo - opponent_elo
    else:
        elo_difference = None

    time_base, time_increment = parse_time_control(
        metadata.get("TimeControl")
    )

    if (
        stockfish_eval_cp is not None
        and previous_eval_cp is not None
    ):
        previous_eval_change_cp = (
            stockfish_eval_cp - previous_eval_cp
        )
    else:
        previous_eval_change_cp = None

    attacked_count, attacked_value = (
        calculate_attacked_pieces(board)
    )

    features = {
        "stockfish_eval_cp": stockfish_eval_cp,

        "material_balance": calculate_material_balance(board),

        "white_piece_count": (
            count_pieces(board, chess.WHITE, chess.KNIGHT)
            + count_pieces(board, chess.WHITE, chess.BISHOP)
            + count_pieces(board, chess.WHITE, chess.ROOK)
            + count_pieces(board, chess.WHITE, chess.QUEEN)
        ),

        "black_piece_count": (
            count_pieces(board, chess.BLACK, chess.KNIGHT)
            + count_pieces(board, chess.BLACK, chess.BISHOP)
            + count_pieces(board, chess.BLACK, chess.ROOK)
            + count_pieces(board, chess.BLACK, chess.QUEEN)
        ),

        "white_pawn_count": count_pieces(
            board, chess.WHITE, chess.PAWN
        ),

        "black_pawn_count": count_pieces(
            board, chess.BLACK, chess.PAWN
        ),

        "legal_check_count": count_legal_checks(board),
        "legal_capture_count": count_legal_captures(board),

        "attacked_piece_count": attacked_count,
        "attacked_piece_value": attacked_value,

        "move_number": board.fullmove_number,
        "game_phase": determine_game_phase(board),

        "previous_eval_change_cp": previous_eval_change_cp,

        "white_elo": white_elo,
        "black_elo": black_elo,
        "elo_difference": elo_difference,

        "time_control_base": time_base,
        "time_control_increment": time_increment,
        "player_time_remaining": player_time_remaining,
    }

    return features
