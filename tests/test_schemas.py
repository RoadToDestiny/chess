"""
Tests for Stage 2 & Stage 3: Dataset schema specification, data contract, and target integrity.
"""

from chess_error_prediction.schemas import (
    METADATA_COLUMNS,
    POSITION_FEATURES,
    META_FEATURES,
    TARGET_COLUMNS,
    ALL_FEATURES,
    ALL_REQUIRED_COLUMNS,
    PositionRecord,
)
from chess_error_prediction.labeling.error_labeler import audit_features_for_leakage


def test_schema_definitions():
    assert "game_id" in METADATA_COLUMNS
    assert "position_id" in METADATA_COLUMNS
    assert "ply" in METADATA_COLUMNS
    assert "eval_before" in POSITION_FEATURES
    assert "player_rating" in META_FEATURES
    assert "eval_loss" in TARGET_COLUMNS
    assert "is_error" in TARGET_COLUMNS

    assert len(ALL_FEATURES) == len(POSITION_FEATURES) + len(META_FEATURES)
    assert len(ALL_REQUIRED_COLUMNS) == len(METADATA_COLUMNS) + len(ALL_FEATURES) + len(TARGET_COLUMNS)


def test_no_leakage_in_features():
    is_clean, violations = audit_features_for_leakage(ALL_FEATURES)
    assert is_clean, f"Target leakage detected in feature space: {violations}"


def test_position_record_structure():
    rec = PositionRecord(
        game_id="g1",
        ply=1,
        move_number=1,
        color="w",
        eval_before=15.0,
        eval_diff_prev_move=0.0,
        material_balance=0.0,
        piece_count=32,
        pawn_count=16,
        legal_checks_count=0,
        legal_captures_count=0,
        attacked_pieces_count=0,
        game_phase=1,
        player_rating=1800.0,
        opponent_rating=1820.0,
        rating_diff=-20.0,
        time_control_base=600.0,
        time_control_inc=0.0,
        eval_after=10.0,
        eval_loss=5.0,
        is_error=0,
    )
    d = rec.to_dict()
    assert d["game_id"] == "g1"
    assert d["position_id"] == "g1_1"
    assert d["is_error"] == 0
    assert d["eval_loss"] == 5.0


def test_features_pov_alignment_and_leakage_detection():
    import chess
    from chess_error_prediction.features.extractors import extract_all_features
    from chess_error_prediction.labeling.error_labeler import audit_features_for_leakage

    # 1. White turn test
    board_w = chess.Board()
    feats_w = extract_all_features(
        board_w,
        eval_before_player_pov=30.0,
        eval_diff_prev_move=10.0,
        white_elo=2000.0,
        black_elo=1850.0,
    )
    assert feats_w["player_rating"] == 2000.0
    assert feats_w["opponent_rating"] == 1850.0
    assert feats_w["rating_diff"] == 150.0

    # 2. Black turn test: 1. e4
    board_b = chess.Board()
    board_b.push_san("e4")
    feats_b = extract_all_features(
        board_b,
        eval_before_player_pov=-30.0,
        eval_diff_prev_move=-10.0,
        white_elo=2000.0,
        black_elo=1850.0,
    )
    assert feats_b["player_rating"] == 1850.0
    assert feats_b["opponent_rating"] == 2000.0
    assert feats_b["rating_diff"] == -150.0

    # 3. Target leakage detection: ensure played_move, result, termination are caught
    leakage_candidates = ["player_rating", "played_move", "game_result", "termination_type"]
    is_clean, violations = audit_features_for_leakage(leakage_candidates)
    assert not is_clean
    assert len(violations) == 3


def test_group_split_protocol():
    from chess_error_prediction.dataset.split import perform_group_split_by_game_id
    mock_games = [f"game_{i}" for i in range(100)]
    splits = perform_group_split_by_game_id(mock_games, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15

    # Check zero overlap
    s_tr = set(splits["train"])
    s_va = set(splits["val"])
    s_te = set(splits["test"])
    assert len(s_tr.intersection(s_va)) == 0
    assert len(s_tr.intersection(s_te)) == 0
    assert len(s_va.intersection(s_te)) == 0
