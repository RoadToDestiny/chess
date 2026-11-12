"""
Stage 6 Feature Audit Script (Backend 3).
Audits extracted features for XGBoost compatibility, range integrity, and zero target leakage.
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
from chess_error_prediction.schemas import ALL_FEATURES, POSITION_FEATURES, META_FEATURES
from chess_error_prediction.features.stage6_feature_verifier import audit_features_for_xgboost


def run_stage6_verification():
    print("=" * 80)
    print("STAGE 6: FEATURE SPACE & TARGET LEAKAGE AUDIT (BACKEND 3)")
    print("=" * 80)

    dataset_path = ROOT_DIR / "data" / "processed" / "final_dataset.csv"
    if not dataset_path.exists():
        print(f"[INFO] {dataset_path} not found. Running audit on labeled sample...")
        dataset_path = ROOT_DIR / "data" / "interim" / "labeled_sample.csv"

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset for feature audit: {dataset_path.name} ({len(df)} rows)")

    # Evaluate all features available in the dataset
    features_to_audit = [f for f in ALL_FEATURES if f in df.columns]
    if not features_to_audit:
        features_to_audit = [c for c in df.columns if c not in ["game_id", "position_id", "fen_before", "move_played", "human_expert_verification", "blunder_category", "is_error", "eval_loss", "eval_after_white"]]

    report = audit_features_for_xgboost(df, features_to_audit)

    print("\n[Feature Audit Report for XGBoost]")
    print(f"   - Status:                  {report['status']}")
    print(f"   - Features Audited:        {report['num_features_evaluated']}")
    print(f"   - Non-numeric Columns:     {report['non_numeric_columns']}")
    print(f"   - Total Nulls:             {report['total_null_entries']}")
    print(f"   - Total Infs:              {report['total_infinite_entries']}")
    print(f"   - Target Leakage Free:     {report['is_leakage_free']}")

    if not report["is_leakage_free"]:
        print(f"   - Leakage Violations:      {report['leakage_violations']}")

    # Save report
    out_file = ROOT_DIR / "reports" / "tables" / "stage6_feature_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Feature audit report saved to {out_file}")


if __name__ == "__main__":
    run_stage6_verification()
