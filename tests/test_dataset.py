"""
Tests for dataset construction, validation checks, and summary generation.
"""

import io
import chess.pgn
import pandas as pd

from chess_error_prediction.dataset.build_dataset import build_dataset_from_game, generate_synthetic_dataset
from chess_error_prediction.dataset.checks import validate_dataset_schema, compute_dataset_summary
from scripts.mwe import mock_evaluator, SAMPLE_PGN


def test_build_dataset_from_single_game():
    game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN))
    assert game is not None

    records = build_dataset_from_game(game, evaluator=mock_evaluator, game_id="test_game_1")
    assert len(records) > 0

    df = pd.DataFrame([r.to_dict() for r in records])
    is_valid, errors = validate_dataset_schema(df)
    assert is_valid, f"Validation errors: {errors}"
    assert "eval_loss" in df.columns
    assert "is_error" in df.columns


def test_synthetic_dataset_conformance():
    df = generate_synthetic_dataset(num_games=10, random_seed=42)
    assert len(df) > 0

    is_valid, errors = validate_dataset_schema(df)
    assert is_valid, f"Synthetic dataset failed schema: {errors}"

    summary = compute_dataset_summary(df)
    assert summary["total_games"] == 10
    assert summary["total_positions"] == len(df)
    assert summary["blunder_count"] >= 0
    assert 0.0 <= summary["blunder_rate"] <= 1.0


def test_dataset_schema_catches_invalid_values():
    df = generate_synthetic_dataset(num_games=5, random_seed=42)
    # Inject bad target
    df.loc[0, "is_error"] = 99
    is_valid, errors = validate_dataset_schema(df)
    assert not is_valid
    assert any("is_error" in e for e in errors)
