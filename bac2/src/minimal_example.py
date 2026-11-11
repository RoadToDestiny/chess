
import io
import chess
import chess.pgn

def build_feature_row(
    game,
    board,
    game_id,
    stockfish_eval_cp,
    target_move,
):
    """
    Преобразует шахматную позицию и сведения о партии
    в одну строку данных.
    
    stockfish_eval_cp пока является тестовой оценкой.
    """

    headers = game.headers

    player_color = "white" if board.turn == chess.WHITE else "black"

    row = {
        "game_id": game_id,
        "position_id": f"{game_id}_ply_{board.ply()}",
        "ply": board.ply(),
        "move_number": board.fullmove_number,
        "player_color": player_color,
        "fen": board.fen(),

        "white_elo": headers.get("WhiteElo"),
        "black_elo": headers.get("BlackElo"),
        "time_control": headers.get("TimeControl"),

        "stockfish_eval_cp": stockfish_eval_cp,

        "target_move_uci": target_move.uci(),
    }

    return row

def run_minimal_example():
    pgn_text = """
[Event "Minimal Example"]
[Site "Local"]
[Date "2026.11.11"]
[Round "1"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]
[TimeControl "600+5"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""

    game = chess.pgn.read_game(io.StringIO(pgn_text))

    if game is None:
        raise ValueError("Не удалось прочитать PGN")

    board = game.board()

    moves = iter(game.mainline_moves())
    target_move = next(moves, None)

    if target_move is None:
        raise ValueError("В партии нет ходов")

    # Тестовая оценка, не результат реального Stockfish.
    test_stockfish_eval_cp = 25

    row = build_feature_row(
        game=game,
        board=board,
        game_id="game_001",
        stockfish_eval_cp=test_stockfish_eval_cp,
        target_move=target_move,
    )

    print("МИНИМАЛЬНЫЙ РАБОЧИЙ ПРИМЕР")
    print("=" * 50)

    for key, value in row.items():
        print(f"{key}: {value}")

    return row

if __name__ == "__main__":
    run_minimal_example()