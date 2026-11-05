"""
Dataset assembly, validation checks, and group splitting package.
"""

from chess_error_prediction.dataset.checks import (
    validate_dataset_schema,
    check_for_target_leakage,
    compute_dataset_summary,
)
from chess_error_prediction.dataset.split import (
    split_by_games,
    verify_split_leakage,
)
from chess_error_prediction.dataset.build_dataset import (
    build_dataset_from_game,
    build_dataset_from_pgn,
    generate_synthetic_dataset,
)

__all__ = [
    "validate_dataset_schema",
    "check_for_target_leakage",
    "compute_dataset_summary",
    "split_by_games",
    "verify_split_leakage",
    "build_dataset_from_game",
    "build_dataset_from_pgn",
    "generate_synthetic_dataset",
]
