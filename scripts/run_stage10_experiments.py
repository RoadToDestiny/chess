"""
Stage 10: Two Additional Experiments for Scientific Paper.
Experiment 1: Three-stage Feature Ablation (Variant A, Variant B, Variant C).
Experiment 2: Threshold Sensitivity Analysis (150 cp, 200 cp, 300 cp).
All evaluations run on the independent hold-out Test split (75 unseen games, 5 046 positions).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from chess_error_prediction.features.feature_spec import (
    FeatureSpec,
    get_default_feature_spec,
    impute_missing_features,
)
from chess_error_prediction.model.metrics import compute_classification_metrics
from chess_error_prediction.model.train import get_default_xgb_params


def run_experiment_1_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    xgb_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Executes Experiment 1: Three-stage Ablation Framework
    - Variant A (11 features): Board / Position only
    - Variant B (16 features): Position + Static Metadata (Ratings + Initial Time Control)
    - Variant C (18 features): Variant B + Dynamic Clock (clock_remaining, time_spent_prev_move)
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: THREE-STAGE FEATURE ABLATION (VARIANTS A, B, C)")
    print("=" * 80)

    variants = [
        ("Variant A (Position Only)", feature_spec.variant_a_features, "11 features (Board-State only)"),
        ("Variant B (Position + Meta)", feature_spec.variant_b_features, "16 features (Board + Ratings + Base Time)"),
        ("Variant C (Position + Meta + Clock)", feature_spec.variant_c_features, "18 features (Board + Meta + Dynamic Clock)"),
    ]

    results = []
    y_train = train_df["is_error"].astype(int).values
    y_val = val_df["is_error"].astype(int).values
    y_test = test_df["is_error"].astype(int).values

    for name, f_cols, desc in variants:
        print(f"\n[Training] {name} - {len(f_cols)} features...")
        X_train = train_df[f_cols].astype(np.float32)
        X_val = val_df[f_cols].astype(np.float32)
        X_test = test_df[f_cols].astype(np.float32)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        prob_val = model.predict_proba(X_val)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]

        m_val = compute_classification_metrics(y_val, prob_val)
        m_test = compute_classification_metrics(y_test, prob_test)

        print(f"   Validation: ROC-AUC={m_val['roc_auc']:.4f}, PR-AUC={m_val['pr_auc']:.4f}, Brier={m_val['brier_score']:.4f}")
        print(f"   Test Set:   ROC-AUC={m_test['roc_auc']:.4f}, PR-AUC={m_test['pr_auc']:.4f}, Brier={m_test['brier_score']:.4f}")

        results.append({
            "variant": name,
            "feature_count": len(f_cols),
            "features_included": desc,
            "val_roc_auc": m_val["roc_auc"],
            "val_pr_auc": m_val["pr_auc"],
            "val_brier_score": m_val["brier_score"],
            "test_roc_auc": m_test["roc_auc"],
            "test_pr_auc": m_test["pr_auc"],
            "test_brier_score": m_test["brier_score"],
            "test_log_loss": m_test["log_loss"],
        })

    return results


def run_experiment_2_thresholds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    xgb_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Executes Experiment 2: Threshold Sensitivity Analysis (150 cp, 200 cp, 300 cp)
    Re-labels blunder target based on eval_loss threshold, keeping asymmetric mate errors.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: THRESHOLD SENSITIVITY ANALYSIS (150, 200, 300 CP)")
    print("=" * 80)

    thresholds = [150.0, 200.0, 300.0]
    results = []

    X_train = train_df[features].astype(np.float32)
    X_val = val_df[features].astype(np.float32)
    X_test = test_df[features].astype(np.float32)

    for tau in thresholds:
        print(f"\n[Threshold tau = {tau:.0f} cp]")

        def get_target(df: pd.DataFrame) -> np.ndarray:
            # y = 1 if eval_loss >= tau or is_mate_error == 1, else 0
            is_err = (df["eval_loss"] >= tau) | (df.get("is_mate_error", 0) == 1)
            return is_err.astype(int).values

        y_train_t = get_target(train_df)
        y_val_t = get_target(val_df)
        y_test_t = get_target(test_df)

        err_rate_train = float(y_train_t.mean() * 100)
        err_rate_test = float(y_test_t.mean() * 100)
        print(f"   Blunder rate -> Train: {err_rate_train:.2f}% | Test: {err_rate_test:.2f}%")

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train,
            y_train_t,
            eval_set=[(X_val, y_val_t)],
            verbose=False,
        )

        prob_val = model.predict_proba(X_val)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]

        m_val = compute_classification_metrics(y_val_t, prob_val)
        m_test = compute_classification_metrics(y_test_t, prob_test)

        print(f"   Validation: ROC-AUC={m_val['roc_auc']:.4f}, PR-AUC={m_val['pr_auc']:.4f}, Brier={m_val['brier_score']:.4f}")
        print(f"   Test Set:   ROC-AUC={m_test['roc_auc']:.4f}, PR-AUC={m_test['pr_auc']:.4f}, Brier={m_test['brier_score']:.4f}")

        results.append({
            "threshold_cp": tau,
            "train_error_rate_pct": round(err_rate_train, 2),
            "test_error_rate_pct": round(err_rate_test, 2),
            "val_roc_auc": m_val["roc_auc"],
            "val_pr_auc": m_val["pr_auc"],
            "val_brier_score": m_val["brier_score"],
            "test_roc_auc": m_test["roc_auc"],
            "test_pr_auc": m_test["pr_auc"],
            "test_brier_score": m_test["brier_score"],
            "test_log_loss": m_test["log_loss"],
        })

    return results


def main():
    splits_dir = ROOT_DIR / "data" / "splits"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    # Impute missing values in dynamic clock features (time_spent on ply 1)
    train_df = impute_missing_features(train_df)
    val_df = impute_missing_features(val_df)
    test_df = impute_missing_features(test_df)

    f_spec = get_default_feature_spec()
    xgb_params = get_default_xgb_params(device="cuda")

    # 1. Run Experiment 1 (Ablation)
    exp1_results = run_experiment_1_ablation(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_spec=f_spec,
        xgb_params=xgb_params,
    )

    # 2. Run Experiment 2 (Thresholds)
    exp2_results = run_experiment_2_thresholds(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        features=f_spec.variant_c_features,  # using complete feature set
        xgb_params=xgb_params,
    )

    tables_dir = ROOT_DIR / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save Table 3: Feature Ablation Results
    df_exp1 = pd.DataFrame(exp1_results)
    table3_path = tables_dir / "table3_feature_ablation_results.csv"
    df_exp1.to_csv(table3_path, index=False)
    print(f"\n[Saved] Table 3 (Feature Ablation) -> {table3_path}")
    print("\nTABLE 3: FEATURE ABLATION COMPARISON (TEST SET):")
    print(df_exp1[["variant", "feature_count", "test_roc_auc", "test_pr_auc", "test_brier_score"]].to_string(index=False))

    # Save Table 4: Threshold Sensitivity Results
    df_exp2 = pd.DataFrame(exp2_results)
    table4_path = tables_dir / "table4_threshold_sensitivity_results.csv"
    df_exp2.to_csv(table4_path, index=False)
    print(f"\n[Saved] Table 4 (Threshold Sensitivity) -> {table4_path}")
    print("\nTABLE 4: THRESHOLD SENSITIVITY COMPARISON (TEST SET):")
    print(df_exp2[["threshold_cp", "test_error_rate_pct", "test_roc_auc", "test_pr_auc", "test_brier_score"]].to_string(index=False))

    print("\n[SUCCESS] Stage 10 Experiments successfully completed.")


if __name__ == "__main__":
    main()
