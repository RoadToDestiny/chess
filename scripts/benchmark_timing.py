"""
Benchmark timing utility (Stage 7 / Stage 5 of Research Plan).
Measures time per position, time per game, and estimates runtime for large-scale dataset generation.
"""

import sys
from pathlib import Path
import time
import io

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chess
import chess.pgn
import pandas as pd

from chess_error_prediction.engine.stockfish_client import StockfishEngine, StockfishError
from chess_error_prediction.dataset.build_dataset import build_dataset_from_game
from scripts.mwe import mock_evaluator, SAMPLE_PGN


def run_benchmark(
    num_games: int = 50,
    output_log: Path = Path("reports/tables/time_benchmark.log"),
):
    output_log = Path(output_log)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"STAGE 7: BENCHMARK TIMING ON {num_games} PILOT GAMES")
    print("=" * 70)

    eval_func = mock_evaluator
    engine_name = "MockEvaluator (CPU fallback)"
    try:
        engine = StockfishEngine()
        engine.start()
        eval_func = engine.analyze
        engine_name = f"Stockfish (Depth {engine.config.depth})"
        print(f"[INFO] Benchmarking with live engine: {engine_name}")
    except (StockfishError, OSError):
        print(f"[INFO] Live engine not available. Benchmarking with: {engine_name}")

    start_time = time.perf_counter()
    total_positions = 0

    for i in range(1, num_games + 1):
        game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN))
        records = build_dataset_from_game(game, evaluator=eval_func, game_id=f"bench_game_{i:03d}")
        total_positions += len(records)

    elapsed = time.perf_counter() - start_time
    sec_per_position = elapsed / max(1, total_positions)
    sec_per_game = elapsed / max(1, num_games)

    est_full_hours = (sec_per_position * 70000.0) / 3600.0

    report = (
        f"====================================================\n"
        f"STOCKFISH & FEATURE EXTRACTION BENCHMARK REPORT\n"
        f"====================================================\n"
        f"Engine used: {engine_name}\n"
        f"Pilot games processed: {num_games}\n"
        f"Total positions analyzed: {total_positions}\n"
        f"Total wall-clock time: {elapsed:.2f} seconds\n"
        f"Average time per position: {sec_per_position:.6f} seconds\n"
        f"Average time per game: {sec_per_game:.4f} seconds\n"
        f"Estimated time for 1,000 games (~70,000 positions): {est_full_hours:.2f} hours\n"
        f"Feasibility status: {'FEASIBLE' if sec_per_position < 2.0 else 'WARNING: DEPTH REDUCTION RECOMMENDED'}\n"
        f"====================================================\n"
    )

    print(report)
    with open(output_log, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[SAVED] Benchmark report written to: {output_log}")


if __name__ == "__main__":
    run_benchmark(num_games=50)
