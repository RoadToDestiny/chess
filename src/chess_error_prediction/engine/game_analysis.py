"""Position-by-position Stockfish analysis for a validated PGN game."""

from dataclasses import dataclass
from typing import Iterator, Optional

import chess
import chess.pgn

from chess_error_prediction.engine.evaluation import EvaluationResult, normalize_evaluation
from chess_error_prediction.engine.stockfish_client import StockfishEngine
from chess_error_prediction.labeling.error_labeler import label_position_evaluations


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
    is_error: int


def analyze_game(
    game: chess.pgn.Game,
    engine: StockfishEngine,
    threshold_cp: float = 200.0,
) -> Iterator[MoveAnalysis]:
    """Yield one analysis record per legal mainline move in ``game``.

    The board is evaluated before and after each move using the same engine
    configuration. Mate evaluations are retained in the record but excluded
    from centipawn-loss labeling.
    """
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move at ply {ply}: {move.uci()}")

        color = "w" if board.turn == chess.WHITE else "b"
        move_number = board.fullmove_number
        san = board.san(move)
        fen_before = board.fen()
        evaluation_before = normalize_evaluation(engine.analyze(board))
        board.push(move)
        fen_after = board.fen()
        evaluation_after = normalize_evaluation(engine.analyze(board))
        label = label_position_evaluations(
            {
                "score_cp": evaluation_before.score_cp,
                "is_mate": evaluation_before.is_mate,
            },
            {
                "score_cp": evaluation_after.score_cp,
                "is_mate": evaluation_after.is_mate,
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
            is_error=label["is_error"],
        )
