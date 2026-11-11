
import chess


REQUIRED_LABEL_FIELDS = {
    "game_id",
    "position_id",
    "ply",
    "player_color",
    "fen",
    "target_move_uci",
    "error",
}


def validate_labeled_record(record):
    """
    Проверяет связь метки ошибки с партией,
    позицией, игроком и целевым ходом.
    """

    missing_fields = REQUIRED_LABEL_FIELDS - record.keys()

    if missing_fields:
        raise ValueError(
            f"Отсутствуют обязательные поля: {missing_fields}"
        )

    if record["error"] not in (0, 1):
        raise ValueError("Метка error должна быть равна 0 или 1")

    if record["player_color"] not in ("white", "black"):
        raise ValueError("Некорректный цвет игрока")

    if not isinstance(record["ply"], int) or record["ply"] < 0:
        raise ValueError("Некорректный номер полухода")

    expected_position_id = (
        f'{record["game_id"]}_ply_{record["ply"]}'
    )

    if record["position_id"] != expected_position_id:
        raise ValueError("position_id не соответствует позиции")

    board = chess.Board(record["fen"])

    if board.ply() != record["ply"]:
        raise ValueError("FEN не соответствует номеру полухода")

    expected_color = "white" if board.turn else "black"

    if record["player_color"] != expected_color:
        raise ValueError("Игрок не соответствует стороне хода в FEN")

    try:
        move = chess.Move.from_uci(record["target_move_uci"])
    except ValueError as exc:
        raise ValueError("Некорректный формат целевого хода") from exc

    if move not in board.legal_moves:
        raise ValueError("Целевой ход нелегален в данной позиции")

    return True
