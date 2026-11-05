"""
Deterministic group-based train/validation/test dataset splitting by whole games.
Guarantees zero data leakage between splits.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def verify_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "game_id"
) -> Tuple[bool, List[str]]:
    """
    Asserts zero game_id intersection across splits.
    """
    train_games = set(train_df[group_col].dropna().unique())
    val_games = set(val_df[group_col].dropna().unique())
    test_games = set(test_df[group_col].dropna().unique())

    errors: List[str] = []
    tv = train_games.intersection(val_games)
    tt = train_games.intersection(test_games)
    vt = val_games.intersection(test_games)

    if tv:
        errors.append(f"{len(tv)} games leaked between train and val!")
    if tt:
        errors.append(f"{len(tt)} games leaked between train and test!")
    if vt:
        errors.append(f"{len(vt)} games leaked between val and test!")

    return len(errors) == 0, errors


def split_by_games(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    group_col: str = "game_id"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs deterministic group-based splitting by whole games (game_id).
    Positions from the same game will NEVER appear in multiple splits.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    unique_games = df[group_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_seed)
    shuffled_games = rng.permutation(unique_games)

    n_total = len(shuffled_games)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_games = set(shuffled_games[:n_train])
    val_games = set(shuffled_games[n_train:n_train + n_val])
    test_games = set(shuffled_games[n_train + n_val:])

    train_df = df[df[group_col].isin(train_games)].copy()
    val_df = df[df[group_col].isin(val_games)].copy()
    test_df = df[df[group_col].isin(test_games)].copy()

    is_clean, errors = verify_split_leakage(train_df, val_df, test_df, group_col=group_col)
    if not is_clean:
        raise RuntimeError(f"Data leakage detected during splitting: {errors}")

    return train_df, val_df, test_df
