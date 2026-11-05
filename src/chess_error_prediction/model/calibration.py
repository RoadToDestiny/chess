"""
Probability calibration diagnostics and feature importance visualization.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
    n_bins: int = 10,
    model_name: str = "XGBoost"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes calibration bins and renders/saves a reliability diagram.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    try:
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(prob_pred, prob_true, marker="s", linewidth=2, color="#1f77b4", label=model_name)
        ax.set_xlabel("Mean Predicted Probability of Error", fontsize=11)
        ax.set_ylabel("Empirical Blunder Fraction", fontsize=11)
        ax.set_title("Probability Calibration Diagram (Reliability Curve)", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

        return prob_true, prob_pred
    except ImportError:
        return np.array([]), np.array([])


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: Optional[Path] = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Renders and saves a horizontal bar chart of feature importance rankings.
    """
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=True)

    if len(df_imp) > top_n:
        df_plot = df_imp.tail(top_n)
    else:
        df_plot = df_imp

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df_plot["feature"], df_plot["importance"], color="#2ca02c")
        ax.set_xlabel("Relative Feature Importance (Gain / Weight)", fontsize=11)
        ax.set_title(f"Top {len(df_plot)} Predictive Features for Chess Blunders", fontsize=12)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
    except ImportError:
        pass

    return df_imp.sort_values(by="importance", ascending=False)
