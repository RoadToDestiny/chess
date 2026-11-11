
import chess

from src.features.extractors import extract_features
from src.features.feature_spec import FEATURE_NAMES


def test_extracts_all_features():
    board = chess.Board()

    metadata = {
        "WhiteElo": "1800",
        "BlackElo": "1750",
        "TimeControl": "600+5",
    }

    features = extract_features(
        board=board,
        metadata=metadata,
        stockfish_eval_cp=25,
        previous_eval_cp=10,
        player_time_remaining=300,
    )

    assert len(features) == len(FEATURE_NAMES)

    assert set(features.keys()) == set(FEATURE_NAMES)


def test_material_balance_in_initial_position():
    board = chess.Board()

    features = extract_features(board, {})

    assert features["material_balance"] == 0


def test_rating_difference_for_white():
    board = chess.Board()

    features = extract_features(
        board,
        {
            "WhiteElo": "1800",
            "BlackElo": "1750",
        },
    )

    assert features["elo_difference"] == 50


def test_time_control_parsing():
    board = chess.Board()

    features = extract_features(
        board,
        {"TimeControl": "600+5"},
    )

    assert features["time_control_base"] == 600
    assert features["time_control_increment"] == 5


def test_missing_data_is_none():
    board = chess.Board()

    features = extract_features(board, {})

    assert features["stockfish_eval_cp"] is None
    assert features["previous_eval_change_cp"] is None
    assert features["white_elo"] is None
    assert features["player_time_remaining"] is None


def test_previous_evaluation_change():
    board = chess.Board()

    features = extract_features(
        board,
        {},
        stockfish_eval_cp=25,
        previous_eval_cp=10,
    )

    assert features["previous_eval_change_cp"] == 15


def test_white_material_advantage():
    board = chess.Board()
    board.remove_piece_at(chess.D8)  # Убираем ферзя чёрных

    features = extract_features(board, {})

    assert features["material_balance"] == 9


def test_black_material_advantage():
    board = chess.Board()
    board.remove_piece_at(chess.D1)  # Убираем ферзя белых

    features = extract_features(board, {})

    assert features["material_balance"] == -9


def test_rating_difference_for_black():
    board = chess.Board()
    board.push_uci("e2e4")  # Теперь ход чёрных

    features = extract_features(
        board,
        {
            "WhiteElo": "1800",
            "BlackElo": "1750",
        },
    )

    assert features["elo_difference"] == -50



def test_attacked_piece_detection():
    board = chess.Board()

    # Чёрная ладья атакует белую ладью по вертикали.
    board.set_fen(
        "4k3/8/8/8/8/8/r7/R3K3 w - - 0 1"
    )

    features = extract_features(board, {})

    assert features["attacked_piece_count"] == 1
    assert features["attacked_piece_value"] == 5



def test_endgame_detection():
    board = chess.Board()
    board.set_fen(
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    )

    features = extract_features(board, {})

    assert features["game_phase"] == 2



def test_features_do_not_depend_on_target_move():
    board = chess.Board()

    metadata = {
        "WhiteElo": "1800",
        "BlackElo": "1750",
        "TimeControl": "600+5",
    }

    # Извлекаем признаки до хода.
    features_before = extract_features(
        board,
        metadata,
        stockfish_eval_cp=25,
        previous_eval_cp=10,
    )

    # Совершаем целевой ход.
    board.push_uci("e2e4")

    # Проверяем, что предыдущая запись не изменилась.
    assert features_before["ply"] if "ply" in features_before else True
    assert features_before["move_number"] == 1
    assert features_before["material_balance"] == 0
    assert features_before["white_pawn_count"] == 8



def test_features_are_numeric_or_none():
    board = chess.Board()

    features = extract_features(
        board,
        {
            "WhiteElo": "1800",
            "BlackElo": "1750",
            "TimeControl": "600+5",
        },
        stockfish_eval_cp=25,
        previous_eval_cp=10,
        player_time_remaining=300,
    )

    for name, value in features.items():
        assert value is None or isinstance(value, (int, float)), (
            f"Признак {name} имеет неподходящий тип: {type(value)}"
        )


def test_target_and_identifiers_are_not_features():
    from src.features.feature_spec import (
        FEATURE_NAMES,
        IDENTIFIER_COLUMNS,
        TARGET_COLUMN,
    )

    assert TARGET_COLUMN not in FEATURE_NAMES

    for column in IDENTIFIER_COLUMNS:
        assert column not in FEATURE_NAMES
