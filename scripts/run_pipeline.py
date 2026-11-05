"""
Unified Research Pipeline Runner.
Orchestrates end-to-end dataset processing, validation, group splitting,
and execution of all 3 scientific experiments.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import json
import yaml
import pandas as pd

from chess_error_prediction.dataset.split import split_by_games
from chess_error_prediction.dataset.checks import (
    validate_dataset_schema,
    check_for_target_leakage,
    compute_dataset_summary,
)
from chess_error_prediction.dataset.build_dataset import generate_synthetic_dataset
from chess_error_prediction.experiments.run_main import run_main_experiment
from chess_error_prediction.experiments.exp1_feature_sets import run_feature_sets_experiment
from chess_error_prediction.experiments.exp2_thresholds import run_threshold_sensitivity_experiment
from chess_error_prediction.schemas import ALL_FEATURES


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_pipeline():
    print("=" * 75)
    print("CHESS BLUNDER PREDICTION: AUTOMATED RESEARCH PIPELINE")
    print("=" * 75)

    cfg = load_config()
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"
    tables_dir = Path("reports/tables")
    figures_dir = Path("reports/figures")

    for d in [processed_dir, splits_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / "final_dataset.csv"

    # Step 1: Load or Generate Dataset
    if dataset_path.exists():
        print(f"\n[Step 1] Loading existing dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print(f"\n[Step 1] Processed dataset not found at {dataset_path}.")
        print("[Step 1] Generating standard research dataset (70 games, ~4,000 positions)...")
        df = generate_synthetic_dataset(
            num_games=70,
            error_threshold_cp=cfg.get("labeling", {}).get("default_threshold_cp", 200.0),
            random_seed=cfg.get("dataset", {}).get("random_seed", 42),
        )
        df.to_csv(dataset_path, index=False)
        print(f"[Step 1] Dataset saved to: {dataset_path}")

    # Step 2: Validate Dataset Schema & Leakage
    print("\n[Step 2] Validating Schema Integrity and Target Leakage...")
    is_valid, schema_errors = validate_dataset_schema(df)
    if not is_valid:
        raise ValueError(f"Dataset schema validation failed: {schema_errors}")
    print("[Step 2] Schema Conformance: PASSED")

    is_clean, leakage_errors = check_for_target_leakage(ALL_FEATURES)
    if not is_clean:
        raise ValueError(f"Target leakage detected in features: {leakage_errors}")
    print("[Step 2] Target Leakage Audit: PASSED (Zero leakage)")

    # Save Descriptive Dataset Summary Table
    summary = compute_dataset_summary(df)
    pd.DataFrame([summary]).to_csv(tables_dir / "dataset_summary.csv", index=False)
    print(f"[Step 2] Dataset Summary: {summary}")

    # Step 3: Perform Group-Based Split (Strictly by game_id)
    print("\n[Step 3] Performing Deterministic Group Split (70/15/15 by game_id)...")
    ds_cfg = cfg.get("dataset", {})
    train_ratio = ds_cfg.get("train_ratio", 0.70)
    val_ratio = ds_cfg.get("val_ratio", 0.15)
    test_ratio = ds_cfg.get("test_ratio", 0.15)
    seed = ds_cfg.get("random_seed", 42)

    train_df, val_df, test_df = split_by_games(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    print(f"[Step 3] Split complete -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Step 4: Run Main Experiment (XGBoost full model)
    print("\n[Step 4] Running Main Experiment (XGBoost on Test Set)...")
    model_params = cfg.get("model", {})
    main_metrics = run_main_experiment(train_df, val_df, test_df, model_params=model_params)
    print(f"[Step 4] Main Experiment Metrics: {main_metrics}")

    # Step 5: Run Experiment 1 (Feature Ablation: Position vs Full)
    print("\n[Step 5] Running Experiment 1 (Ablation: Positional vs Full Feature Sets)...")
    exp1_df = run_feature_sets_experiment(train_df, val_df, test_df, model_params=model_params)
    print(exp1_df.to_string(index=False))

    # Step 6: Run Experiment 2 (Threshold Sensitivity: 150, 200, 300 cp)
    print("\n[Step 6] Running Experiment 2 (Threshold Sensitivity: 150, 200, 300 cp)...")
    thresholds = cfg.get("labeling", {}).get("sensitivity_thresholds", [150.0, 200.0, 300.0])
    exp2_df = run_threshold_sensitivity_experiment(train_df, val_df, test_df, thresholds=thresholds, model_params=model_params)
    print(exp2_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("[COMPLETE] ALL RESEARCH EXPERIMENTS FINISHED SUCCESSFULLY.")
    print("Outputs:")
    print("  - Models:     artifacts/models/model.json")
    print("  - Runs:       artifacts/runs/main_run.json")
    print("  - Tables:     reports/tables/ (*.csv)")
    print("  - Figures:    reports/figures/ (*.png)")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
