"""
Tests for centipawn loss computation, blunder labeling, and mate filtering.
"""

import pandas as pd
from chess_error_prediction.labeling.error_labeler import (
    compute_centipawn_loss,
    assign_blunder_label,
    label_position_evaluations,
    filter_mate_records,
)


def test_centipawn_loss_calculation():
    # White moves: eval drops from +200 to -50 -> loss is 250 cp
    loss_w = compute_centipawn_loss(200.0, -50.0, color="w")
    assert loss_w == 250.0

    # White makes good move: eval increases from +50 to +150 -> negative loss -100
    loss_w_good = compute_centipawn_loss(50.0, 150.0, color="w")
    assert loss_w_good == -100.0

    # Black moves: eval from White POV rises from -100 to +150 -> Black made a blunder, loss is 250 cp
    loss_b = compute_centipawn_loss(-100.0, 150.0, color="b")
    assert loss_b == 250.0


def test_blunder_threshold_labeling():
    assert assign_blunder_label(199.9, threshold_cp=200.0) == 0
    assert assign_blunder_label(200.0, threshold_cp=200.0) == 1
    assert assign_blunder_label(350.0, threshold_cp=200.0) == 1

    # Sensitivity thresholds
    assert assign_blunder_label(170.0, threshold_cp=150.0) == 1
    assert assign_blunder_label(170.0, threshold_cp=200.0) == 0
    assert assign_blunder_label(250.0, threshold_cp=300.0) == 0


def test_label_position_evaluations_dict_integration():
    eval_before = {"score_cp": 50.0, "is_mate": False, "mate_in": None, "depth": 18}
    eval_after = {"score_cp": -200.0, "is_mate": False, "mate_in": None, "depth": 18}

    res = label_position_evaluations(eval_before, eval_after, color="w", threshold_cp=200.0)
    assert res["eval_loss"] == 250.0
    assert res["is_error"] == 1
    assert res["is_mate_eval"] is False


def test_asymmetric_mate_blunder_labeling():
    # 1. White blunders into checkmate: eval_before was normal +100 cp, eval_after Black mates in 2 (mate_in = -2)
    eval_before_normal = {"score_cp": 100.0, "is_mate": False, "mate_in": None, "depth": 18}
    eval_after_mate_black = {"score_cp": -100000.0, "is_mate": True, "mate_in": -2, "depth": 18}
    res_w_blunder = label_position_evaluations(eval_before_normal, eval_after_mate_black, color="w")
    assert res_w_blunder["is_error"] == 1
    assert res_w_blunder["eval_loss"] == 2500.0
    assert res_w_blunder["is_mate_eval"] is False

    # 2. White delivers checkmate: eval_after White mates in 1 (mate_in = +1)
    eval_after_mate_white = {"score_cp": 100000.0, "is_mate": True, "mate_in": 1, "depth": 18}
    res_w_win = label_position_evaluations(eval_before_normal, eval_after_mate_white, color="w")
    assert res_w_win["is_error"] == 0
    assert res_w_win["eval_loss"] == 0.0
    assert res_w_win["is_mate_eval"] is False

    # 3. Position before move was already forced mate: excluded with is_mate_eval = True
    eval_before_already_mate = {"score_cp": -100000.0, "is_mate": True, "mate_in": -1, "depth": 18}
    res_already_mate = label_position_evaluations(eval_before_already_mate, eval_after_mate_black, color="w")
    assert res_already_mate["is_mate_eval"] is True
    assert res_already_mate["is_error"] == 0
    assert res_already_mate["eval_loss"] is None


def test_mate_filtering():
    df = pd.DataFrame({
        "game_id": ["g1", "g1", "g2"],
        "is_mate_eval": [False, True, False],
        "eval_loss": [10.0, 9999.0, 50.0],
    })
    filtered, stats = filter_mate_records(df)
    assert len(filtered) == 2
    assert stats["excluded"] == 1
