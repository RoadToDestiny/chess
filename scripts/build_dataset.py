"""Build the labeled feature dataset and benchmark report from filtered PGN."""

import argparse
import json
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import chess

from chess_error_prediction.data.pgn_reader import (
    extract_clock_features,
    open_pgn_file,
    parse_time_control,
    read_games,
)
from chess_error_prediction.dataset.benchmark import BenchmarkMetrics
from chess_error_prediction.engine.game_analysis import analyze_game
from chess_error_prediction.engine.stockfish_client import StockfishEngine
from chess_error_prediction.features.extractors import extract_all_features
from chess_error_prediction.labeling.stage5_target_verifier import (
    analyze_threshold_sensitivity_distribution,
)


def build_dataset(input_path: Path, output_path: Path, benchmark_path: Path) -> None:
    rows = []
    metrics = BenchmarkMetrics()
    tracemalloc.start()
    dataset_started = perf_counter()

    with open_pgn_file(input_path) as source, StockfishEngine() as engine:
        games = iter(read_games(source))
        game_index = 0
        while True:
            metrics.start_stage("pgn")
            try:
                game = next(games)
            except StopIteration:
                metrics.stop_stage("pgn")
                break
            metrics.stop_stage("pgn")
            game_index += 1
            metrics.games_read += 1
            metadata = dict(game.headers)
            game_id = metadata.get("Site", "").rstrip("/").split("/")[-1] or f"game_{game_index}"
            base_seconds, increment_seconds = parse_time_control(metadata.get("TimeControl", ""))
            clocks = extract_clock_features(game, base_seconds, increment_seconds)
            board = game.board()
            previous_eval_white = None

            for record in analyze_game(game, engine, benchmark=metrics):
                current_eval_white = record.evaluation_before.score_cp
                if record.eval_loss is None:
                    previous_eval_white = current_eval_white
                    board.push(chess.Move.from_uci(record.uci))
                    continue

                if previous_eval_white is None:
                    eval_diff = 0.0
                elif record.color == "w":
                    eval_diff = current_eval_white - previous_eval_white
                else:
                    eval_diff = previous_eval_white - current_eval_white

                metrics.start_stage("features")
                features = extract_all_features(
                    board,
                    eval_before_player_pov=(
                        current_eval_white
                        if record.color == "w"
                        else -current_eval_white
                    ),
                    eval_diff_prev_move=eval_diff,
                    white_elo=float(metadata.get("WhiteElo", 1500)),
                    black_elo=float(metadata.get("BlackElo", 1500)),
                    time_control_base=base_seconds,
                    time_control_inc=increment_seconds,
                    clock_remaining=clocks.get(record.ply, {}).get("clock_remaining"),
                    time_spent_prev_move=clocks.get(record.ply, {}).get("time_spent_prev_move"),
                    has_prev_move=record.ply > 1,
                )
                metrics.stop_stage("features")
                features.update(
                    {
                        "game_id": game_id,
                        "position_id": f"{game_id}_{record.ply}",
                        "ply": record.ply,
                        "color": record.color,
                        "fen": record.fen_before,
                        "eval_after": record.evaluation_after.score_cp,
                        "eval_loss": record.eval_loss,
                        "is_error": record.is_error,
                        "is_mate_eval": record.is_mate_eval,
                        "is_mate_error": record.is_mate_error,
                    }
                )
                rows.append(features)
                previous_eval_white = current_eval_white
                board.push(chess.Move.from_uci(record.uci))
    metrics.dataset_seconds = perf_counter() - dataset_started
    metrics.refresh_peak_memory()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = pd.DataFrame(rows)
    dataset.to_csv(output_path, index=False)
    reports_dir = ROOT_DIR / "reports" / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)
    analyze_threshold_sensitivity_distribution(
        dataset["eval_loss"] if not dataset.empty else pd.Series(dtype=float)
    ).to_csv(reports_dir / "stage5_threshold_distribution.csv", index=False)
    split_rows = []
    for split_name in ("train", "val", "test"):
        split_path = ROOT_DIR / "data" / "splits" / f"{split_name}_game_ids.csv"
        if not split_path.exists():
            continue
        split_ids = set(pd.read_csv(split_path)["game_id"].astype(str))
        split_df = dataset[dataset["game_id"].astype(str).isin(split_ids)]
        split_rows.append(
            {
                "split": split_name,
                "games": len(split_ids),
                "valid_rows": len(split_df),
                "ordinary_moves": int((split_df["is_error"] == 0).sum()),
                "errors": int((split_df["is_error"] == 1).sum()),
                "error_rate_pct": round(float(split_df["is_error"].mean() * 100), 2)
                if not split_df.empty
                else 0.0,
            }
        )
    pd.DataFrame(split_rows).to_csv(reports_dir / "dataset_row_counts.csv", index=False)
    benchmark_path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    tracemalloc.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT_DIR / "data/raw/filtered_games.pgn")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "data/processed/final_dataset.csv")
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "reports/tables/benchmark.json")
    args = parser.parse_args()
    build_dataset(args.input, args.output, args.benchmark)


if __name__ == "__main__":
    main()
