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


# ------------------ STAGE 1 TESTS (BACKEND 3) ------------------

def test_stage1_xgboost_gpu_acceleration():
    """Verify GPU-accelerated training (NVIDIA CUDA) with XGBoost on tabular features."""
    import numpy as np
    import xgboost as xgb
    X = np.random.randn(25, len(ALL_FEATURES)).astype(np.float32)
    y = np.random.randint(0, 2, 25).astype(np.int32)
    clf = xgb.XGBClassifier(n_estimators=3, tree_method="hist", device="cuda", random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (25, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


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
    # Pre-move mate is isolated
    pre_mate_res = verify_mate_isolation_protocol(is_mate_before=True, is_mate_after=True)
    assert pre_mate_res["is_mate_eval"] is True
    assert pre_mate_res["is_mate_error"] is False
    assert pre_mate_res["exclude_from_cp_loss"] is True

    # Post-move blunder into mate is preserved as blunder
    post_mate_res = verify_mate_isolation_protocol(is_mate_before=False, is_mate_after=True)
    assert post_mate_res["is_mate_eval"] is False
    assert post_mate_res["is_mate_error"] is True
    assert post_mate_res["exclude_from_cp_loss"] is False


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
        "has_prev_move": [1, 1],
        "material_balance": [0.0, 1.0],
        "piece_count": [14, 13],
        "pawn_count": [8, 7],
        "legal_checks_count": [0, 1],
        "legal_captures_count": [1, 2],
        "attacked_pieces_count": [0, 1],
        "move_number": [1, 11],
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


def test_grouped_calibration_uses_oof_predictions():
    from chess_error_prediction.model.calibration import (
        evaluate_calibrators,
        fit_grouped_calibrators,
    )
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({"signal": np.linspace(-2.0, 2.0, 24)})
    y = (X["signal"].to_numpy() > 0).astype(int)
    groups = np.repeat(np.arange(8), 3)
    result = fit_grouped_calibrators(
        X,
        y,
        groups,
        estimator_factory=lambda: LogisticRegression(max_iter=500),
        n_splits=4,
    )

    assert result.fold_count == 4
    assert np.isfinite(result.oof_probability).all()
    metrics = evaluate_calibrators(result, X, y)
    assert set(metrics) == {"uncalibrated", "platt", "isotonic"}


def test_logistic_regression_pipeline_spec_and_leakage_free_fit():
    from chess_error_prediction.model.baseline_lr import (
        create_logistic_regression_pipeline,
        get_lr_feature_lists,
    )

    numeric_cols, categorical_cols, dropped_cols = get_lr_feature_lists(ALL_FEATURES)
    assert "rating_diff" in dropped_cols
    assert "game_phase" in categorical_cols
    assert "move_number" in numeric_cols
    assert "has_prev_move" in numeric_cols

    # Synthesize small dataset
    n_samples = 30
    df = pd.DataFrame({
        "eval_before": np.random.randn(n_samples),
        "eval_diff_prev_move": np.random.randn(n_samples),
        "has_prev_move": np.random.choice([0, 1], n_samples),
        "material_balance": np.random.randn(n_samples),
        "piece_count": np.random.randint(4, 30, n_samples),
        "pawn_count": np.random.randint(2, 16, n_samples),
        "legal_checks_count": np.random.randint(0, 3, n_samples),
        "legal_captures_count": np.random.randint(0, 5, n_samples),
        "attacked_pieces_count": np.random.randint(0, 4, n_samples),
        "move_number": np.random.randint(1, 50, n_samples),
        "game_phase": np.random.choice([1, 2, 3], n_samples),
        "player_rating": np.random.uniform(1600, 2400, n_samples),
        "opponent_rating": np.random.uniform(1600, 2400, n_samples),
        "rating_diff": np.zeros(n_samples),  # Dependent collinear column
        "time_control_base": np.full(n_samples, 600.0),
        "time_control_inc": np.zeros(n_samples),
    })
    y = np.random.randint(0, 2, n_samples)

    pipeline = create_logistic_regression_pipeline(ALL_FEATURES)
    pipeline.fit(df, y)
    probs = pipeline.predict_proba(df)
    assert probs.shape == (n_samples, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

