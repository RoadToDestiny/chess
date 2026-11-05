"""
Tests for group-based train/validation/test dataset splitting.
"""

import pytest
import pandas as pd
from chess_error_prediction.dataset.split import split_by_games, verify_split_leakage
from chess_error_prediction.dataset.build_dataset import generate_synthetic_dataset


def test_group_split_zero_leakage():
    df = generate_synthetic_dataset(num_games=25, random_seed=42)
    train_df, val_df, test_df = split_by_games(df, 0.70, 0.15, 0.15, random_seed=42)

    is_clean, errors = verify_split_leakage(train_df, val_df, test_df)
    assert is_clean, f"Data leakage between splits: {errors}"

    train_games = set(train_df["game_id"].unique())
    val_games = set(val_df["game_id"].unique())
    test_games = set(test_df["game_id"].unique())

    # Assert zero overlap
    assert len(train_games.intersection(val_games)) == 0
    assert len(train_games.intersection(test_games)) == 0
    assert len(val_games.intersection(test_games)) == 0

    # Total rows sum to original
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_invalid_ratios_raise_error():
    df = generate_synthetic_dataset(num_games=5, random_seed=42)
    with pytest.raises(ValueError):
        split_by_games(df, 0.80, 0.15, 0.15)  # Sums to 1.10
