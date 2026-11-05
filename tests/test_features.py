"""
Tests for chess feature extraction and metadata calculation.
"""

import chess
from chess_error_prediction.features.extractors import extract_board_features, extract_all_features
from chess_error_prediction.features.feature_spec import FeatureSpec, get_default_feature_spec, impute_missing_features
from chess_error_prediction.dataset.checks import check_for_target_leakage


def test_board_features_initial_position():
    board = chess.Board()
    feats = extract_board_features(board)

    assert feats["pawn_count"] == 16
    assert feats["piece_count"] == 14  # 4 knights + 4 bishops + 4 rooks + 2 queens
    assert feats["material_balance"] == 0.0
    assert feats["legal_checks_count"] == 0
    assert feats["legal_captures_count"] == 0
    assert feats["game_phase"] == 1  # Opening moves <= 10


def test_board_features_check_and_attack():
    board = chess.Board()
    # Scholar's mate attack line: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
    for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]:
        board.push_san(board.san(chess.Move.from_uci(uci)))

    feats = extract_board_features(board)
    assert feats["legal_checks_count"] >= 1  # Qxf7+ is legal
    assert feats["legal_captures_count"] >= 1


def test_extract_all_features_black_perspective():
    board = chess.Board()
    board.push_san("e4")  # Now it's Black's turn

    feats = extract_all_features(
        board=board,
        eval_before_player_pov=-30.0,
        eval_diff_prev_move=30.0,
        white_elo=2000.0,
        black_elo=1950.0,
        time_control_base=600.0,
        time_control_inc=5.0,
    )

    assert feats["player_rating"] == 1950.0
    assert feats["opponent_rating"] == 2000.0
    assert feats["rating_diff"] == -50.0
    assert feats["time_control_base"] == 600.0
    assert feats["time_control_inc"] == 5.0


def test_feature_leakage_audit():
    spec = get_default_feature_spec()
    is_clean, violations = check_for_target_leakage(spec.all_features)
    assert is_clean, f"Target leakage detected: {violations}"
