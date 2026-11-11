
import io

import chess.pgn


def test_position_changes_after_move():
    pgn_text = """
[Event "Position test"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None

    board = game.board()
    initial_fen = board.fen()

    moves = iter(game.mainline_moves())
    board.push(next(moves))

    assert board.fen() != initial_fen
    assert board.ply() == 1
    assert board.turn == chess.BLACK
