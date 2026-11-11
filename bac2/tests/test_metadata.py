
import io

import chess.pgn


def test_ratings_and_time_control_are_read():
    pgn_text = """
[Event "Rated Blitz"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1450"]
[TimeControl "180+2"]

1. e4 e5 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    assert game is not None
    assert game.headers["WhiteElo"] == "1500"
    assert game.headers["BlackElo"] == "1450"
    assert game.headers["TimeControl"] == "180+2"


def test_missing_metadata_can_be_detected():
    pgn_text = """
[Event "No metadata"]
[White "Player A"]
[Black "Player B"]
[Result "*"]

1. d4 d5 *
"""
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    assert game is not None
    assert "WhiteElo" not in game.headers
    assert "BlackElo" not in game.headers
    assert "TimeControl" not in game.headers
