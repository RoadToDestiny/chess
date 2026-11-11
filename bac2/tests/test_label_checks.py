
import chess

import pytest

from src.dataset.checks import validate_labeled_record


def create_valid_record():
    board = chess.Board()

    return {
        "game_id": "game_001",
        "position_id": "game_001_ply_0",
        "ply": 0,
        "player_color": "white",
        "fen": board.fen(),
        "target_move_uci": "e2e4",
        "error": 1,
    }


def test_valid_labeled_record():
    record = create_valid_record()

    assert validate_labeled_record(record) is True


def test_invalid_error_label():
    record = create_valid_record()
    record["error"] = 2

    with pytest.raises(ValueError):
        validate_labeled_record(record)


def test_wrong_player_color():
    record = create_valid_record()
    record["player_color"] = "black"

    with pytest.raises(ValueError):
        validate_labeled_record(record)


def test_wrong_position_id():
    record = create_valid_record()
    record["position_id"] = "game_002_ply_0"

    with pytest.raises(ValueError):
        validate_labeled_record(record)


def test_illegal_target_move():
    record = create_valid_record()
    record["target_move_uci"] = "e2e5"

    with pytest.raises(ValueError):
        validate_labeled_record(record)


def test_missing_required_field():
    record = create_valid_record()
    del record["game_id"]

    with pytest.raises(ValueError):
        validate_labeled_record(record)
