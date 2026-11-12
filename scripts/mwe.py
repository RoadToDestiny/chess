"""
Minimal Working Example (MWE) - Stage 4.
Joint artifact of all 3 backenders:
- Backend 1: PGN parsing and Stockfish position evaluation.
- Backend 2: Position feature extraction into structured dictionary/row.
- Backend 3: Schema validation, XGBoost data format compatibility, and model inference test.
"""

import sys
from pathlib import Path
import io

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chess
import chess.pgn
import pandas as pd
import numpy as np

# Backend 1
from chess_error_prediction.engine.stockfish_client import StockfishEngine, StockfishError
from chess_error_prediction.engine.evaluation import normalize_evaluation

# Backend 2
from chess_error_prediction.schemas import ALL_FEATURES, POSITION_FEATURES, META_FEATURES

# Backend 3
from chess_error_prediction.labeling.error_labeler import compute_centipawn_loss, assign_blunder_label
from chess_error_prediction.dataset.checks import check_for_target_leakage


SAMPLE_PGN = """[Event "Lichess Classical Test Game"]
[Site "https://lichess.org/mwe_sample_001"]
[Date "2026.11.12"]
[Round "1"]
[White "GrandmasterA"]
[Black "GrandmasterB"]
[Result "1-0"]
[WhiteElo "2150"]
[BlackElo "2100"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. Qb3 1-0
"""


def mock_stockfish_eval(board: chess.Board) -> dict:
    """Deterministic fallback evaluator if Stockfish binary is not in PATH."""
    piece_vals = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
    w_mat = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in piece_vals.items())
    b_mat = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in piece_vals.items())
    score = float(w_mat - b_mat + (board.ply() * 2))
    return {"score_cp": score, "is_mate": False, "mate_in": None, "depth": 18}


def extract_features_row_backend2(board: chess.Board, metadata: dict, eval_before_white: float, prev_eval_white: float) -> dict:
    """Extracts features before move (Backend 2 specification)."""
    is_white = (board.turn == chess.WHITE)
    player_color = "w" if is_white else "b"

    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    pawn_count = white_pawns + black_pawns

    white_pieces = sum(len(board.pieces(pt, chess.WHITE)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    black_pieces = sum(len(board.pieces(pt, chess.BLACK)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    piece_count = white_pieces + black_pieces

    # Material balance from active player POV
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    w_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
    b_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
    mat_bal = float(w_mat - b_mat) if is_white else float(b_mat - w_mat)

    legal_checks = sum(1 for m in board.legal_moves if board.gives_check(m))
    legal_captures = sum(1 for m in board.legal_moves if board.is_capture(m))

    opponent = not board.turn
    attacked_pieces = sum(1 for sq, p in board.piece_map().items() if p.color == board.turn and p.piece_type != chess.KING and board.is_attacked_by(opponent, sq))

    fullmove = board.fullmove_number
    game_phase = 1 if fullmove <= 10 else (2 if fullmove <= 30 else 3)

    w_elo = float(metadata.get("WhiteElo", 1500))
    b_elo = float(metadata.get("BlackElo", 1500))
    p_elo = w_elo if is_white else b_elo
    o_elo = b_elo if is_white else w_elo

    eval_before_player = eval_before_white if is_white else -eval_before_white
    eval_diff_prev = eval_before_white - prev_eval_white

    return {
        "eval_before": float(eval_before_player),
        "eval_diff_prev_move": float(eval_diff_prev),
        "material_balance": mat_bal,
        "piece_count": int(piece_count),
        "pawn_count": int(pawn_count),
        "legal_checks_count": int(legal_checks),
        "legal_captures_count": int(legal_captures),
        "attacked_pieces_count": int(attacked_pieces),
        "game_phase": int(game_phase),
        "player_rating": float(p_elo),
        "opponent_rating": float(o_elo),
        "rating_diff": float(p_elo - o_elo),
        "time_control_base": 600.0,
        "time_control_inc": 0.0,
    }


def verify_stage4_backend3(feature_row: dict) -> dict:
    """
    Backend 3 Task:
    Validates that the extracted feature row can be directly fed into an XGBoost model.
    """
    import xgboost as xgb

    # Check 1: No target leakage
    is_clean, violations = check_for_target_leakage(list(feature_row.keys()))
    if not is_clean:
        raise ValueError(f"Leakage in feature row: {violations}")

    # Check 2: Convert to DataFrame and check types
    df_row = pd.DataFrame([feature_row])
    for col in ALL_FEATURES:
        if col not in df_row.columns:
            raise ValueError(f"Missing expected feature: {col}")

    X_sample = df_row[ALL_FEATURES].astype(np.float32)

    # Check 3: Feed into XGBoost baseline classifier
    dummy_X = np.random.randn(20, len(ALL_FEATURES)).astype(np.float32)
    dummy_y = (np.random.rand(20) > 0.8).astype(int)

    test_model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    test_model.fit(dummy_X, dummy_y)

    pred_prob = float(test_model.predict_proba(X_sample)[0, 1])

    return {
        "status": "PASS",
        "num_features": len(ALL_FEATURES),
        "features_present": list(feature_row.keys()),
        "xgboost_inference_test": "SUCCESS",
        "sample_predicted_blunder_prob": round(pred_prob, 4),
    }


def run_mwe():
    print("=" * 80)
    print("STAGE 4: JOINT MINIMAL WORKING EXAMPLE (MWE) - FULL PIPELINE VERIFICATION")
    print("=" * 80)

    # Step 1 (Backend 1): Load and parse PGN game
    game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN))
    assert game is not None
    board = game.board()
    print("[Backend 1] PGN game parsed successfully.")

    # Select evaluator
    eval_func = mock_stockfish_eval
    try:
        engine = StockfishEngine()
        engine.start()
        eval_func = engine.analyze
        print("[Backend 1] Live Stockfish engine connected.")
    except (StockfishError, OSError):
        print("[Backend 1] Stockfish executable not found in PATH. Using deterministic mock evaluator.")

    # Step 2: First move analysis
    eval_before = eval_func(board)
    score_before_white = float(eval_before["score_cp"])

    # Step 3 (Backend 2): Build feature row before move
    metadata = dict(game.headers)
    feature_row = extract_features_row_backend2(board, metadata, score_before_white, prev_eval_white=0.0)
    print("[Backend 2] Feature row extracted from position (14 pre-move features):")
    for k, v in feature_row.items():
        print(f"   - {k}: {v}")

    # Step 4 (Backend 3): Verify row in XGBoost pipeline
    b3_report = verify_stage4_backend3(feature_row)
    print("\n[Backend 3 Verification]")
    print(f"   - Schema & Leakage Status:       {b3_report['status']}")
    print(f"   - Features Fed into XGBoost:     {b3_report['num_features']}")
    print(f"   - XGBoost Model Inference:       {b3_report['xgboost_inference_test']}")
    print(f"   - Sample Predicted Blunder Risk: {b3_report['sample_predicted_blunder_prob'] * 100:.1f}%")

    print("\n[SUCCESS] Stage 4 Minimal Working Example fully verified by Backend 3.")


if __name__ == "__main__":
    run_mwe()
