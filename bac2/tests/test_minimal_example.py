
import io

import chess.pgn

from src.minimal_example import build_feature_row


def test_build_feature_row():
    pgn_text = """
[Event "Test"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]
[TimeControl "600+5"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None

    board = game.board()
    target_move = next(iter(game.mainline_moves()))

    row = build_feature_row(
        game=game,
        board=board,
        game_id="game_001",
        stockfish_eval_cp=25,
        target_move=target_move,
    )

    assert row["game_id"] == "game_001"
    assert row["position_id"] == "game_001_ply_0"
    assert row["ply"] == 0
    assert row["player_color"] == "white"
    assert row["white_elo"] == "1800"
    assert row["black_elo"] == "1750"
    assert row["time_control"] == "600+5"
    assert row["stockfish_eval_cp"] == 25
    assert row["target_move_uci"] == "e2e4"

    # Позиция должна соответствовать состоянию ДО хода.
    assert row["fen"] == board.fen()
