"""
Stage 9: Main Experiment & Comprehensive Quality Evaluation.
Evaluates XGBoost vs LogisticRegression on the untouched Test split (75 unseen games).
Generates:
- Comparative metrics table (ROC-AUC, PR-AUC, Brier Score, Log Loss)
- Calibration curves (Reliability Diagram)
- ROC curves and PR curves
- Cluster Bootstrap confidence intervals across independent games
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from chess_error_prediction.schemas import ALL_FEATURES
from chess_error_prediction.model import (
    train_models,
    compute_classification_metrics,
    save_training_artifacts,
)


def compute_cluster_bootstrap_ci(
    df_test: pd.DataFrame,
    y_prob_xgb: np.ndarray,
    y_prob_lr: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Computes 95% Cluster Bootstrap confidence intervals resampling by independent game_id.
    """
    rng = np.random.default_rng(random_state)
    unique_games = df_test["game_id"].unique()
    n_games = len(unique_games)

    xgb_roc, xgb_pr, xgb_brier = [], [], []
    lr_roc, lr_pr, lr_brier = [], [], []

    # Map game_id to indices
    game_indices = {g: np.where(df_test["game_id"].values == g)[0] for g in unique_games}
    y_test_all = df_test["is_error"].values

    for _ in range(n_bootstrap):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        sampled_idx = np.concatenate([game_indices[g] for g in sampled_games])

        y_boot = y_test_all[sampled_idx]
        if len(np.unique(y_boot)) < 2:
            continue

        p_xgb = y_prob_xgb[sampled_idx]
        p_lr = y_prob_lr[sampled_idx]

        m_xgb = compute_classification_metrics(y_boot, p_xgb)
        m_lr = compute_classification_metrics(y_boot, p_lr)

        xgb_roc.append(m_xgb["roc_auc"])
        xgb_pr.append(m_xgb["pr_auc"])
        xgb_brier.append(m_xgb["brier_score"])

        lr_roc.append(m_lr["roc_auc"])
        lr_pr.append(m_lr["pr_auc"])
        lr_brier.append(m_lr["brier_score"])

    def get_ci(arr):
        return (round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4))

    return {
        "xgboost": {
            "roc_auc_ci": get_ci(xgb_roc),
            "pr_auc_ci": get_ci(xgb_pr),
            "brier_ci": get_ci(xgb_brier),
        },
        "logistic_regression": {
            "roc_auc_ci": get_ci(lr_roc),
            "pr_auc_ci": get_ci(lr_pr),
            "brier_ci": get_ci(lr_brier),
        },
    }


def run_stage9_main_experiment():
    print("=" * 80)
    print("STAGE 9: MAIN EXPERIMENT - XGBOOST VS LOGISTIC REGRESSION BASELINE")
    print("=" * 80)

    splits_dir = ROOT_DIR / "data" / "splits"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    print(f"Loaded datasets:")
    print(f"   Train: {len(train_df)} observations ({train_df['game_id'].nunique()} games)")
    print(f"   Val:   {len(val_df)} observations ({val_df['game_id'].nunique()} games)")
    print(f"   Test:  {len(test_df)} observations ({test_df['game_id'].nunique()} games)")

    # 1. Train models on Train, tune on Val, test on Test
    print("\n[Step 1] Fitting XGBoost and LogisticRegression...")
    res = train_models(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=ALL_FEATURES,
        target_col="is_error",
        group_col="game_id",
        enable_calibration=True,
        device="cpu",
    )

    # Persist the exact model and metrics produced by this Stage 9 run.
    models_dir = ROOT_DIR / "artifacts" / "models"
    runs_dir = ROOT_DIR / "artifacts" / "runs"
    model_path, metadata_path, run_path = save_training_artifacts(
        result=res,
        models_dir=models_dir,
        runs_dir=runs_dir,
    )
    print(f"[Artifacts] Model saved to: {model_path}")
    print(f"[Artifacts] Model metadata saved to: {metadata_path}")
    print(f"[Artifacts] Run metadata saved to: {run_path}")

    X_test = test_df[ALL_FEATURES].astype(np.float32)
    y_test = test_df["is_error"].values

    xgb_prob = res.xgb_model.predict_proba(X_test)[:, 1]
    lr_prob = res.lr_pipeline.predict_proba(X_test)[:, 1]

    # Calibrated probabilities
    xgb_iso_prob = res.xgb_calibrators.isotonic.transform(xgb_prob) if res.xgb_calibrators else xgb_prob
    xgb_platt_prob = res.xgb_calibrators.platt.predict_proba(xgb_prob.reshape(-1, 1))[:, 1] if res.xgb_calibrators else xgb_prob

    # 2. Compute metrics
    m_xgb = compute_classification_metrics(y_test, xgb_prob)
    m_xgb_iso = compute_classification_metrics(y_test, xgb_iso_prob)
    m_xgb_platt = compute_classification_metrics(y_test, xgb_platt_prob)
    m_lr = compute_classification_metrics(y_test, lr_prob)

    # 3. Compute 95% Cluster Bootstrap CIs
    print("\n[Step 2] Computing 95% Cluster Bootstrap Confidence Intervals (1000 resamples by game_id)...")
    boot_ci = compute_cluster_bootstrap_ci(test_df, xgb_prob, lr_prob, n_bootstrap=1000)

    print("\n" + "=" * 80)
    print("TABLE 2: MAIN EXPERIMENT EVALUATION ON TEST SET (75 UNSEEN GAMES)")
    print("=" * 80)
    print(f"{'Model':<25} | {'ROC-AUC [95% CI]':<22} | {'PR-AUC [95% CI]':<22} | {'Brier Score [95% CI]':<24} | {'Log Loss':<8}")
    print("-" * 110)
    print(f"{'LogisticRegression (Base)':<25} | {m_lr['roc_auc']:.4f} {str(boot_ci['logistic_regression']['roc_auc_ci']):<15} | "
          f"{m_lr['pr_auc']:.4f} {str(boot_ci['logistic_regression']['pr_auc_ci']):<15} | "
          f"{m_lr['brier_score']:.4f} {str(boot_ci['logistic_regression']['brier_ci']):<17} | {m_lr['log_loss']:.4f}")
    print(f"{'XGBoost (Raw)':<25} | {m_xgb['roc_auc']:.4f} {str(boot_ci['xgboost']['roc_auc_ci']):<15} | "
          f"{m_xgb['pr_auc']:.4f} {str(boot_ci['xgboost']['pr_auc_ci']):<15} | "
          f"{m_xgb['brier_score']:.4f} {str(boot_ci['xgboost']['brier_ci']):<17} | {m_xgb['log_loss']:.4f}")
    print(f"{'XGBoost (Platt Cal)':<25} | {m_xgb_platt['roc_auc']:.4f} {'[Calibrated]':<15} | "
          f"{m_xgb_platt['pr_auc']:.4f} {'[Calibrated]':<15} | "
          f"{m_xgb_platt['brier_score']:.4f} {'[Calibrated]':<17} | {m_xgb_platt['log_loss']:.4f}")
    print(f"{'XGBoost (Isotonic Cal)':<25} | {m_xgb_iso['roc_auc']:.4f} {'[Calibrated]':<15} | "
          f"{m_xgb_iso['pr_auc']:.4f} {'[Calibrated]':<15} | "
          f"{m_xgb_iso['brier_score']:.4f} {'[Calibrated]':<17} | {m_xgb_iso['log_loss']:.4f}")
    print("=" * 110)

    # 4. Save Table 2 to CSV and JSON
    tables_dir = ROOT_DIR / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    table2_rows = [
        {
            "model": "LogisticRegression",
            "variant": "Baseline (L2, Scaled)",
            "roc_auc": m_lr["roc_auc"],
            "roc_auc_ci_lower": boot_ci["logistic_regression"]["roc_auc_ci"][0],
            "roc_auc_ci_upper": boot_ci["logistic_regression"]["roc_auc_ci"][1],
            "pr_auc": m_lr["pr_auc"],
            "pr_auc_ci_lower": boot_ci["logistic_regression"]["pr_auc_ci"][0],
            "pr_auc_ci_upper": boot_ci["logistic_regression"]["pr_auc_ci"][1],
            "brier_score": m_lr["brier_score"],
            "brier_ci_lower": boot_ci["logistic_regression"]["brier_ci"][0],
            "brier_ci_upper": boot_ci["logistic_regression"]["brier_ci"][1],
            "log_loss": m_lr["log_loss"],
        },
        {
            "model": "XGBoost",
            "variant": "Raw Probabilities",
            "roc_auc": m_xgb["roc_auc"],
            "roc_auc_ci_lower": boot_ci["xgboost"]["roc_auc_ci"][0],
            "roc_auc_ci_upper": boot_ci["xgboost"]["roc_auc_ci"][1],
            "pr_auc": m_xgb["pr_auc"],
            "pr_auc_ci_lower": boot_ci["xgboost"]["pr_auc_ci"][0],
            "pr_auc_ci_upper": boot_ci["xgboost"]["pr_auc_ci"][1],
            "brier_score": m_xgb["brier_score"],
            "brier_ci_lower": boot_ci["xgboost"]["brier_ci"][0],
            "brier_ci_upper": boot_ci["xgboost"]["brier_ci"][1],
            "log_loss": m_xgb["log_loss"],
        },
        {
            "model": "XGBoost",
            "variant": "Platt Calibrated",
            "roc_auc": m_xgb_platt["roc_auc"],
            "roc_auc_ci_lower": np.nan,
            "roc_auc_ci_upper": np.nan,
            "pr_auc": m_xgb_platt["pr_auc"],
            "pr_auc_ci_lower": np.nan,
            "pr_auc_ci_upper": np.nan,
            "brier_score": m_xgb_platt["brier_score"],
            "brier_ci_lower": np.nan,
            "brier_ci_upper": np.nan,
            "log_loss": m_xgb_platt["log_loss"],
        },
        {
            "model": "XGBoost",
            "variant": "Isotonic Calibrated",
            "roc_auc": m_xgb_iso["roc_auc"],
            "roc_auc_ci_lower": np.nan,
            "roc_auc_ci_upper": np.nan,
            "pr_auc": m_xgb_iso["pr_auc"],
            "pr_auc_ci_lower": np.nan,
            "pr_auc_ci_upper": np.nan,
            "brier_score": m_xgb_iso["brier_score"],
            "brier_ci_lower": np.nan,
            "brier_ci_upper": np.nan,
            "log_loss": m_xgb_iso["log_loss"],
        },
    ]

    pd.DataFrame(table2_rows).to_csv(tables_dir / "table2_main_experiment_results.csv", index=False)
    print(f"\n[Artifacts] Table 2 saved to: {tables_dir / 'table2_main_experiment_results.csv'}")

    # 5. Generate Figures (Scientific Plots)
    figures_dir = ROOT_DIR / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Calibration Diagram (Reliability Curve)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    prob_true_lr, prob_pred_lr = calibration_curve(y_test, lr_prob, n_bins=10, strategy="uniform")
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, xgb_prob, n_bins=10, strategy="uniform")
    prob_true_iso, prob_pred_iso = calibration_curve(y_test, xgb_iso_prob, n_bins=10, strategy="uniform")

    ax.plot(prob_pred_lr, prob_true_lr, "s-", color="#1f77b4", label=f"Logistic Regression (Brier={m_lr['brier_score']:.4f})")
    ax.plot(prob_pred_xgb, prob_true_xgb, "^-", color="#ff7f0e", label=f"XGBoost Raw (Brier={m_xgb['brier_score']:.4f})")
    ax.plot(prob_pred_iso, prob_true_iso, "o-", color="#2ca02c", label=f"XGBoost Isotonic (Brier={m_xgb_iso['brier_score']:.4f})")

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Empirical Blunder Rate)", fontsize=11)
    ax.set_title("Figure 1: Probability Calibration Diagram (Test Set, 75 Games)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    fig1_path = figures_dir / "figure1_calibration_diagram.png"
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"[Artifacts] Figure 1 (Calibration) saved to: {fig1_path}")

    # Plot 2: ROC and Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax1.plot(fpr_lr, tpr_lr, color="#1f77b4", label=f"Logistic Regression (AUC = {m_lr['roc_auc']:.4f})")
    ax1.plot(fpr_xgb, tpr_xgb, color="#d62728", lw=2, label=f"XGBoost (AUC = {m_xgb['roc_auc']:.4f})")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax1.set_title("ROC Curves", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)

    # PR Curves
    rec_lr, prec_lr, _ = precision_recall_curve(y_test, lr_prob)
    rec_xgb, prec_xgb, _ = precision_recall_curve(y_test, xgb_prob)
    baseline_rate = y_test.mean()

    ax2.plot([0, 1], [baseline_rate, baseline_rate], "k--", label=f"Random Guess ({baseline_rate:.1%})", alpha=0.6)
    ax2.plot(rec_lr, prec_lr, color="#1f77b4", label=f"Logistic Regression (PR-AUC = {m_lr['pr_auc']:.4f})")
    ax2.plot(rec_xgb, prec_xgb, color="#d62728", lw=2, label=f"XGBoost (PR-AUC = {m_xgb['pr_auc']:.4f})")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle("Figure 2: Discrimination Ability on Hold-out Test Set (5 046 positions)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2_path = figures_dir / "figure2_roc_pr_curves.png"
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"[Artifacts] Figure 2 (ROC & PR) saved to: {fig2_path}")

    # Plot 3: Feature Importance (Gain & Weight)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    booster = res.xgb_model.get_booster()
    importance_gain = booster.get_score(importance_type="gain")
    # Sort features by gain
    sorted_feats = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)
    f_names = [x[0] for x in sorted_feats][::-1]
    f_scores = [x[1] for x in sorted_feats][::-1]

    y_pos = np.arange(len(f_names))
    ax.barh(y_pos, f_scores, color="#2b5c8f", align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(f_names, fontsize=10)
    ax.set_xlabel("Average Gain per Split (XGBoost)", fontsize=11)
    ax.set_title("Figure 3: Feature Importance Hierarchy (Gain)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3_path = figures_dir / "figure3_feature_importance.png"
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"[Artifacts] Figure 3 (Feature Importance) saved to: {fig3_path}")

    print("\n[SUCCESS] Stage 9 Main Experiment fully executed.")


if __name__ == "__main__":
    run_stage9_main_experiment()
