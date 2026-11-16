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

    import chess
    from chess_error_prediction.features.extractors import extract_all_features

    dataset_path = ROOT_DIR / "data" / "processed" / "final_dataset.csv"
    if not dataset_path.exists():
        print(f"[INFO] {dataset_path} not found. Extracting complete research feature set for labeled sample...")
        sample_path = ROOT_DIR / "data" / "interim" / "labeled_sample.csv"
        df_raw = pd.read_csv(sample_path)
        rows = []
        for _, r in df_raw.iterrows():
            board = chess.Board(r["fen_before"])
            ev_white = float(r["eval_before_white"])
            ev_player = ev_white if board.turn == chess.WHITE else -ev_white
            feats = extract_all_features(
                board,
                eval_before_player_pov=ev_player,
                eval_diff_prev_move=0.0,
                white_elo=2000.0,
                black_elo=2000.0,
                has_prev_move=bool(r["ply"] > 1),
            )
            rows.append(feats)
        df = pd.DataFrame(rows)
        print(f"Constructed audit DataFrame with {len(df)} positions and {len(ALL_FEATURES)} features.")
    else:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset for feature audit: {dataset_path.name} ({len(df)} rows)")

    # Evaluate all 16 features from research specification
    features_to_audit = ALL_FEATURES

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
