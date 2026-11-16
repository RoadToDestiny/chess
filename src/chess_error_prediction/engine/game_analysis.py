"""Position-by-position Stockfish analysis for a validated PGN game."""

from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Optional

import chess
import chess.pgn

from chess_error_prediction.engine.evaluation import EvaluationResult, normalize_evaluation
from chess_error_prediction.engine.stockfish_client import StockfishEngine
from chess_error_prediction.data.pgn_reader import extract_clock_features, parse_time_control
from chess_error_prediction.labeling.error_labeler import label_position_evaluations
from chess_error_prediction.dataset.benchmark import BenchmarkMetrics


@dataclass(frozen=True)
class MoveAnalysis:
    """Engine evaluations and the error label for one played move."""

    ply: int
    move_number: int
    color: str
    san: str
    uci: str
    fen_before: str
    fen_after: str
    evaluation_before: EvaluationResult
    evaluation_after: EvaluationResult
    eval_loss: Optional[float]
    is_mate_eval: bool
    is_mate_error: bool
    clock_remaining: Optional[float]
    time_spent_prev_move: Optional[float]
    is_error: int


def analyze_game(
    game: chess.pgn.Game,
    engine: StockfishEngine,
    threshold_cp: float = 200.0,
    benchmark: Optional[BenchmarkMetrics] = None,
) -> Iterator[MoveAnalysis]:
    """Yield one analysis record per legal mainline move in ``game``.

    The board is evaluated before and after each move using the same engine
    configuration. To reduce computational complexity by 50%, evaluation after
    ply k is cached and reused as evaluation before ply k+1.
    Mate evaluations are retained in the record but excluded from centipawn-loss labeling.
    """
    board = game.board()
    started = perf_counter()
    current_eval = normalize_evaluation(engine.analyze(board))
    if benchmark is not None:
        benchmark.record_stockfish_call(perf_counter() - started)
    try:
        base_seconds, increment_seconds = parse_time_control(game.headers.get("TimeControl", ""))
        clock_features = extract_clock_features(game, base_seconds, increment_seconds)
    except ValueError:
        clock_features = {}

    for ply, move in enumerate(game.mainline_moves(), start=1):
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move at ply {ply}: {move.uci()}")

        color = "w" if board.turn == chess.WHITE else "b"
        move_number = board.fullmove_number
        san = board.san(move)
        fen_before = board.fen()
        evaluation_before = current_eval

        board.push(move)
        fen_after = board.fen()
        started = perf_counter()
        evaluation_after = normalize_evaluation(engine.analyze(board))
        if benchmark is not None:
            benchmark.record_stockfish_call(perf_counter() - started)
        current_eval = evaluation_after

        label = label_position_evaluations(
            {
                "score_cp": evaluation_before.score_cp,
                "is_mate": evaluation_before.is_mate,
                "mate_in": evaluation_before.mate_in,
            },
            {
                "score_cp": evaluation_after.score_cp,
                "is_mate": evaluation_after.is_mate,
                "mate_in": evaluation_after.mate_in,
            },
            color=color,
            threshold_cp=threshold_cp,
        )

        yield MoveAnalysis(
            ply=ply,
            move_number=move_number,
            color=color,
            san=san,
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=fen_after,
            evaluation_before=evaluation_before,
            evaluation_after=evaluation_after,
            eval_loss=label["eval_loss"],
            is_mate_eval=label["is_mate_eval"],
            is_mate_error=label["is_mate_error"],
            clock_remaining=clock_features.get(ply, {}).get("clock_remaining"),
            time_spent_prev_move=clock_features.get(ply, {}).get("time_spent_prev_move"),
            is_error=label["is_error"],
        )
        if benchmark is not None:
            benchmark.moves_read += 1
            if label["is_mate_eval"]:
                benchmark.excluded_pre_move_mates += 1
            elif label["is_mate_error"]:
                benchmark.saved_post_move_mate_errors += 1
            elif label["eval_loss"] is None:
                benchmark.excluded_other_terminal += 1
            else:
                benchmark.valid_rows += 1
