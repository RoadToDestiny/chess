"""
Stage 5 Verification Script (Backend 3).
Audits labeled_sample.csv, verifies binary target variable integrity,
and computes class balance statistics and recommended scale_pos_weight for XGBoost.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
from chess_error_prediction.labeling.stage5_target_verifier import (
    verify_binary_target_integrity,
    analyze_threshold_sensitivity_distribution,
)


def run_stage5_verification():
    print("=" * 80)
    print("STAGE 5: TARGET VARIABLE INTEGRITY & CLASS BALANCE AUDIT (BACKEND 3)")
    print("=" * 80)

    sample_path = ROOT_DIR / "data" / "interim" / "labeled_sample.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {sample_path}")

    df = pd.read_csv(sample_path)
    print(f"Loaded labeled sample with {len(df)} positions.")

    # 1. Target integrity check
    report = verify_binary_target_integrity(df["is_error"])
    print("\n[Target Integrity Report]")
    for k, v in report.items():
        print(f"   - {k}: {v}")

    if report["status"] != "PASS":
        raise ValueError("Target validation FAILED!")

    # 2. Threshold sensitivity distribution check
    sens_df = analyze_threshold_sensitivity_distribution(df["eval_loss"])
    print("\n[Threshold Sensitivity Distribution (150, 200, 300 cp)]")
    print(sens_df.to_string(index=False))

    # Save summary report
    out_dir = ROOT_DIR / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    sens_df.to_csv(out_dir / "stage5_threshold_distribution.csv", index=False)
    print(f"\n[SUCCESS] Stage 5 verification completed. Output saved to {out_dir / 'stage5_threshold_distribution.csv'}")


if __name__ == "__main__":
    run_stage5_verification()
