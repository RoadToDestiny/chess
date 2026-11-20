"""
Group-based dataset splitting protocol (70/15/15) by game_id.
Guarantees zero data leakage between train, validation, and test sets.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def perform_group_split_by_game_id(
    game_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Partitions unique game_ids into train, val, and test splits.
    Strictly asserts that no game_id appears in more than one partition.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    unique_ids = sorted(list(set(game_ids)))
    n_games = len(unique_ids)
    if n_games == 0:
        raise ValueError("Cannot split empty list of game_ids.")

    rng = np.random.RandomState(random_seed)
    shuffled = rng.permutation(unique_ids)

    n_train = int(round(train_ratio * n_games))
    n_val = int(round(val_ratio * n_games))

    train_ids = sorted(list(shuffled[:n_train]))
    val_ids = sorted(list(shuffled[n_train : n_train + n_val]))
    test_ids = sorted(list(shuffled[n_train + n_val :]))

    # Strict zero-leakage assertions
    s_train: Set[str] = set(train_ids)
    s_val: Set[str] = set(val_ids)
    s_test: Set[str] = set(test_ids)

    assert len(s_train.intersection(s_val)) == 0, "Train and Val splits share game_ids!"
    assert len(s_train.intersection(s_test)) == 0, "Train and Test splits share game_ids!"
    assert len(s_val.intersection(s_test)) == 0, "Val and Test splits share game_ids!"
    assert len(s_train) + len(s_val) + len(s_test) == n_games, "Game ID count mismatch after split!"

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def save_split_ids(
    splits: Dict[str, List[str]],
    splits_dir: Path = Path("data/splits"),
) -> Dict[str, Path]:
    """
    Saves partitioned game IDs into permanent CSV files for reproducibility.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, ids in splits.items():
        file_path = splits_dir / f"{name}_game_ids.csv"
        df = pd.DataFrame({"game_id": ids})
        df.to_csv(file_path, index=False)
        paths[name] = file_path

    return paths
