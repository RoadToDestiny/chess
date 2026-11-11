
import io
import chess.pgn
import pandas as pd

pgn_text = """
[Event "Rated Blitz game"]
[Site "Test"]
[Date "2026.10.23"]
[Round "1"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1450"]
[TimeControl "180+2"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Game without ratings"]
[Site "Test"]
[Date "2026.10.23"]
[Round "2"]
[White "Player C"]
[Black "Player D"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
"""

stream = io.StringIO(pgn_text)
records = []

while True:
    game = chess.pgn.read_game(stream)

    if game is None:
        break

    headers = game.headers

    records.append({
        "event": headers.get("Event"),
        "white": headers.get("White"),
        "black": headers.get("Black"),
        "white_elo": headers.get("WhiteElo"),
        "black_elo": headers.get("BlackElo"),
        "time_control": headers.get("TimeControl"),
        "result": headers.get("Result"),
        "plies": game.end().ply(),
    })

df = pd.DataFrame(records)

print("Метаданные PGN:")
print(df.to_string(index=False))

print("\nПроверка пропусков:")
print(df.isna().sum())

print("\nВсего партий:", len(df))
