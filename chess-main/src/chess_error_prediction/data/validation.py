"""
Game validation for PGN integrity and complete move legality.
"""

from typing import List, Optional, Tuple
import chess
import chess.pgn


REQUIRED_PGN_HEADERS: Tuple[str, ...] = (
    "Event",
    "Site",
    "Date",
    "Round",
    "White",
    "Black",
    "Result",
    "WhiteElo",
    "BlackElo",
    "TimeControl",
)


def validate_game(game: chess.pgn.Game) -> Tuple[bool, List[str]]:
    """
    Validates a single chess game:
    - Presence of all required headers
    - No parser errors
    - All mainline moves are strictly legal
    """
    reasons: List[str] = []

    # Check required headers
    for header in REQUIRED_PGN_HEADERS:
        val = game.headers.get(header)
        if not val or not str(val).strip() or str(val).strip() == "?":
            reasons.append(f"Missing or invalid required header: '{header}'")

    # Check parser errors
    if game.errors:
        reasons.append(f"Game contains {len(game.errors)} PGN parsing errors: {game.errors}")

    # Check legal moves
    board = game.board()
    try:
        for move_idx, move in enumerate(game.mainline_moves(), start=1):
            if move not in board.legal_moves:
                reasons.append(f"Illegal move encountered at ply {move_idx}: {move.uci()}")
                break
            board.push(move)
    except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError) as exc:
        reasons.append(f"Move traversal error: {exc}")

    return len(reasons) == 0, reasons
