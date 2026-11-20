"""
Stockfish engine management and evaluation normalization.
"""

from chess_error_prediction.engine.stockfish_client import StockfishConfig, StockfishEngine, StockfishError
from chess_error_prediction.engine.evaluation import EvaluationResult, normalize_evaluation

__all__ = [
    "StockfishConfig",
    "StockfishEngine",
    "StockfishError",
    "EvaluationResult",
    "normalize_evaluation",
]
from chess_error_prediction.engine.game_analysis import MoveAnalysis, analyze_game
from chess_error_prediction.engine.stockfish_client import (
    StockfishConfig,
    StockfishEngine,
    StockfishError,
)

__all__ = [
    "MoveAnalysis",
    "StockfishConfig",
    "StockfishEngine",
    "StockfishError",
    "analyze_game",
]
