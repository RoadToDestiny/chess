
import io

import chess.pgn


def test_pgn_can_be_read():
    pgn_text = """
[Event "Test Game"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    assert game is not None
    assert game.headers["White"] == "Player A"
    assert game.headers["Black"] == "Player B"
    assert game.headers["Result"] == "1-0"
    assert game.end().ply() == 4
