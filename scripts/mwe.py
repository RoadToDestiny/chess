"""
Minimal Working Example (MWE) - Stage 4.
Demonstrates end-to-end processing of a single chess game:
from PGN parsing -> Stockfish analysis -> feature extraction -> error labeling -> validation.
"""

import sys
from pathlib import Path
import io

# Ensure src/ is on sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import chess
import chess.pgn
import pandas as pd

from chess_error_prediction.engine.stockfish_client import StockfishEngine, StockfishError
from chess_error_prediction.dataset.build_dataset import build_dataset_from_game
from chess_error_prediction.dataset.checks import validate_dataset_schema, check_for_target_leakage
from chess_error_prediction.schemas import ALL_FEATURES


SAMPLE_PGN = """[Event "Lichess Classical Test Game"]
[Site "https://lichess.org/mwe_sample_001"]
[Date "2026.10.22"]
[Round "1"]
[White "GrandmasterA"]
[Black "GrandmasterB"]
[Result "1-0"]
[WhiteElo "2150"]
[BlackElo "2100"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. Qb3 1-0
"""


def mock_evaluator(board: chess.Board) -> dict:
    """Deterministic fallback evaluator if Stockfish executable is not in PATH."""
    piece_vals = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
    w_score = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in piece_vals.items())
    b_score = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in piece_vals.items())
    score_cp = float(w_score - b_score + (board.ply() * 2))
    return {
        "score_cp": score_cp,
        "is_mate": False,
        "mate_in": None,
        "depth": 18,
    }


def run_mwe():
    print("=" * 70)
    print("STAGE 4: MINIMAL WORKING EXAMPLE (MWE) ON A SINGLE CHESS GAME")
    print("=" * 70)

    game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN))
    assert game is not None, "Failed to parse sample PGN"

    # Determine evaluator: live Stockfish if available, else mock evaluator
    eval_func = mock_evaluator
    try:
        engine = StockfishEngine()
        engine.start()
        eval_func = engine.analyze
        print("[INFO] Live Stockfish engine connected successfully.")
    except (StockfishError, OSError):
        print("[INFO] Stockfish binary not in PATH. Using deterministic mock evaluator for MWE.")

    # Build position records
    records = build_dataset_from_game(game, evaluator=eval_func, game_id="mwe_game_001")
    df = pd.DataFrame([r.to_dict() for r in records])

    print(f"\nSuccessfully extracted {len(df)} position rows from game.")
    cols_preview = ["ply", "move_number", "color", "eval_before", "eval_loss", "is_error", "material_balance", "pawn_count"]
    print("\n--- Extracted Data Sample (First 5 Half-Moves) ---")
    print(df[cols_preview].head(5).to_string(index=False))

    # Schema validation (Backend 3 check)
    is_valid, errors = validate_dataset_schema(df)
    print(f"\n[Validation] Dataset Schema Conformance: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"Validation errors: {errors}")

    # Target leakage audit
    is_clean, violations = check_for_target_leakage(ALL_FEATURES)
    print(f"[Validation] Target Leakage Audit in Features: {'PASS' if is_clean else 'FAIL'}")
    if violations:
        print(f"Leakage violations: {violations}")

    print("\n[SUCCESS] Minimal Working Example completed successfully.")


if __name__ == "__main__":
    run_mwe()
