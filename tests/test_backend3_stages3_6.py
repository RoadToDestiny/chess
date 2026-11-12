"""
Unit and integration tests for Backend 3 tasks on Stages 3, 4, 5, 6.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Stage 3
from chess_error_prediction.engine.stage3_eval_verifier import (
    verify_evaluation_scale_and_perspective,
    verify_mate_isolation_protocol,
    generate_env_setup_log,
)

# Stage 4
from scripts.mwe import verify_stage4_backend3, extract_features_row_backend2
import chess

# Stage 5
from chess_error_prediction.labeling.stage5_target_verifier import (
    verify_binary_target_integrity,
    analyze_threshold_sensitivity_distribution,
)

# Stage 6
from chess_error_prediction.features.stage6_feature_verifier import (
    audit_target_leakage,
    audit_features_for_xgboost,
)
from chess_error_prediction.schemas import ALL_FEATURES


# ------------------ STAGE 3 TESTS ------------------

def test_stage3_evaluation_perspective_white_and_black():
    # White moves: eval drops from +150 to -70 -> White made a blunder of 220 cp
    res_w = verify_evaluation_scale_and_perspective(150.0, -70.0, turn_color="w", threshold_cp=200.0)
    assert res_w["eval_loss_player_pov"] == 220.0
    assert res_w["is_error"] == 1
    assert res_w["is_blunder_detected"] is True

    # Black moves: eval from White POV rises from -100 to +120 -> Black made a blunder of 220 cp
    res_b = verify_evaluation_scale_and_perspective(-100.0, 120.0, turn_color="b", threshold_cp=200.0)
    assert res_b["eval_loss_player_pov"] == 220.0
    assert res_b["is_error"] == 1

    # Non-blunder: White moves, eval drops by only 50 cp
    res_non = verify_evaluation_scale_and_perspective(100.0, 50.0, turn_color="w", threshold_cp=200.0)
    assert res_non["eval_loss_player_pov"] == 50.0
    assert res_non["is_error"] == 0


def test_stage3_mate_isolation_protocol():
    mate_res = verify_mate_isolation_protocol(is_mate_before=False, is_mate_after=True)
    assert mate_res["is_mate_eval"] is True
    assert mate_res["exclude_from_cp_loss"] is True


def test_stage3_env_setup_log_artifact(tmp_path: Path):
    log_file = tmp_path / "env_setup.log"
    content = generate_env_setup_log(str(log_file), stockfish_version="Stockfish 17", depth=18)
    assert log_file.exists()
    assert "Stockfish 17" in content
    assert "18 plies" in content


# ------------------ STAGE 4 TESTS ------------------

def test_stage4_mwe_backend3_inference_pipeline():
    board = chess.Board()
    row = extract_features_row_backend2(board, {"WhiteElo": 2000, "BlackElo": 1900}, eval_before_white=20.0, prev_eval_white=0.0)

    report = verify_stage4_backend3(row)
    assert report["status"] == "PASS"
    assert report["xgboost_inference_test"] == "SUCCESS"
    assert 0.0 <= report["sample_predicted_blunder_prob"] <= 1.0


# ------------------ STAGE 5 TESTS ------------------

def test_stage5_binary_target_integrity():
    y_valid = pd.Series([0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    report = verify_binary_target_integrity(y_valid)
    assert report["status"] == "PASS"
    assert report["is_strictly_binary"] is True
    assert report["positive_count_blunders"] == 3
    assert report["negative_count_non_blunders"] == 7
    assert report["recommended_scale_pos_weight"] == pytest.approx(2.33, 0.05)


def test_stage5_invalid_target_caught():
    y_invalid = pd.Series([0, 1, 2, 0])
    report = verify_binary_target_integrity(y_invalid)
    assert report["status"] == "FAIL"
    assert report["is_strictly_binary"] is False


def test_stage5_threshold_distribution():
    eval_losses = pd.Series([10.0, 50.0, 160.0, 210.0, 350.0])
    dist_df = analyze_threshold_sensitivity_distribution(eval_losses, thresholds=[150.0, 200.0, 300.0])
    assert len(dist_df) == 3
    assert dist_df.loc[dist_df["threshold_cp"] == 150.0, "blunder_count"].values[0] == 3
    assert dist_df.loc[dist_df["threshold_cp"] == 200.0, "blunder_count"].values[0] == 2
    assert dist_df.loc[dist_df["threshold_cp"] == 300.0, "blunder_count"].values[0] == 1


# ------------------ STAGE 6 TESTS ------------------

def test_stage6_leakage_detection():
    clean_features = ALL_FEATURES
    is_clean, violations = audit_target_leakage(clean_features)
    assert is_clean, f"Target leakage in clean features: {violations}"

    leaky_features = ["eval_before", "eval_after", "eval_loss", "is_error", "move_played"]
    is_clean, violations = audit_target_leakage(leaky_features)
    assert not is_clean
    assert len(violations) >= 3


def test_stage6_xgboost_compatibility_audit():
    df_clean = pd.DataFrame({
        "eval_before": [10.0, -20.0],
        "eval_diff_prev_move": [5.0, -10.0],
        "material_balance": [0.0, 1.0],
        "piece_count": [14, 13],
        "pawn_count": [8, 7],
        "legal_checks_count": [0, 1],
        "legal_captures_count": [1, 2],
        "attacked_pieces_count": [0, 1],
        "game_phase": [1, 2],
        "player_rating": [1800.0, 1850.0],
        "opponent_rating": [1820.0, 1800.0],
        "rating_diff": [-20.0, 50.0],
        "time_control_base": [600.0, 600.0],
        "time_control_inc": [0.0, 0.0],
    })
    report = audit_features_for_xgboost(df_clean, ALL_FEATURES)
    assert report["status"] == "PASS"
    assert report["total_null_entries"] == 0
    assert report["is_leakage_free"] is True
