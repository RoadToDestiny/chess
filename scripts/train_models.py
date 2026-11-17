"""
Verification script for Stage 8: Model Training (XGBoost & LogisticRegression).
Executes end-to-end model training, probability calibration, and saves artifacts.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
from chess_error_prediction.schemas import ALL_FEATURES
from chess_error_prediction.model import train_models, save_training_artifacts


def main():
    print("=" * 80)
    print("STAGE 8: MODEL TRAINING (XGBOOST & LOGISTIC REGRESSION BASELINE)")
    print("=" * 80)

    # 1. Load data splits
    train_path = ROOT_DIR / "data" / "splits" / "train.csv"
    val_path = ROOT_DIR / "data" / "splits" / "val.csv"
    test_path = ROOT_DIR / "data" / "splits" / "test.csv"

    print(f"[Data] Loading splits from {train_path.parent}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"   - Train rows: {len(train_df)}, Blunder rate: {train_df['is_error'].mean():.2%}")
    print(f"   - Val rows:   {len(val_df)}, Blunder rate: {val_df['is_error'].mean():.2%}")
    print(f"   - Test rows:  {len(test_df)}, Blunder rate: {test_df['is_error'].mean():.2%}")

    # Ensure all required features are present
    for df, name in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    # 2. Train models
    print("\n[Training] Training XGBoost and LogisticRegression...")
    res = train_models(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=ALL_FEATURES,
        target_col="is_error",
        group_col="game_id",
        enable_calibration=True,
        device="cuda",
    )

    print("\n[Evaluation] Validation Set Metrics:")
    print(f"   XGBoost:             ROC-AUC={res.metrics_val['xgboost']['roc_auc']:.4f}, "
          f"PR-AUC={res.metrics_val['xgboost']['pr_auc']:.4f}, "
          f"Brier={res.metrics_val['xgboost']['brier_score']:.4f}")
    print(f"   Logistic Regression: ROC-AUC={res.metrics_val['logistic_regression']['roc_auc']:.4f}, "
          f"PR-AUC={res.metrics_val['logistic_regression']['pr_auc']:.4f}, "
          f"Brier={res.metrics_val['logistic_regression']['brier_score']:.4f}")

    print("\n[Evaluation] Test Set Metrics (Unseen Games):")
    print(f"   XGBoost (Raw):       ROC-AUC={res.metrics_test['xgboost']['roc_auc']:.4f}, "
          f"PR-AUC={res.metrics_test['xgboost']['pr_auc']:.4f}, "
          f"Brier={res.metrics_test['xgboost']['brier_score']:.4f}")
    if "xgboost_platt" in res.metrics_test:
        print(f"   XGBoost (Platt Cal): ROC-AUC={res.metrics_test['xgboost_platt']['roc_auc']:.4f}, "
              f"PR-AUC={res.metrics_test['xgboost_platt']['pr_auc']:.4f}, "
              f"Brier={res.metrics_test['xgboost_platt']['brier_score']:.4f}")
    if "xgboost_isotonic" in res.metrics_test:
        print(f"   XGBoost (Isotonic):  ROC-AUC={res.metrics_test['xgboost_isotonic']['roc_auc']:.4f}, "
              f"PR-AUC={res.metrics_test['xgboost_isotonic']['pr_auc']:.4f}, "
              f"Brier={res.metrics_test['xgboost_isotonic']['brier_score']:.4f}")
    print(f"   Logistic Regression: ROC-AUC={res.metrics_test['logistic_regression']['roc_auc']:.4f}, "
          f"PR-AUC={res.metrics_test['logistic_regression']['pr_auc']:.4f}, "
          f"Brier={res.metrics_test['logistic_regression']['brier_score']:.4f}")

    # 3. Save artifacts
    models_dir = ROOT_DIR / "artifacts" / "models"
    runs_dir = ROOT_DIR / "artifacts" / "runs"
    m_path, meta_path, r_path = save_training_artifacts(res, models_dir, runs_dir)
    print(f"\n[Artifacts] Saved model to: {m_path}")
    print(f"[Artifacts] Saved metadata to: {meta_path}")
    print(f"[Artifacts] Saved run info to: {r_path}")

    print("\n[SUCCESS] Stage 8 Model Training completed successfully.")


if __name__ == "__main__":
    main()
