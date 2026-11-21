import sys
sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from chess_error_prediction.features.feature_spec import FeatureSpec

ROOT_DIR = Path(__file__).resolve().parent.parent
figures_dir = ROOT_DIR / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(ROOT_DIR / "data" / "splits" / "train.csv")
val_df = pd.read_csv(ROOT_DIR / "data" / "splits" / "val.csv")

# 1. Load Variant B from artifacts/models/model.json
booster_b = xgb.Booster()
booster_b.load_model(str(ROOT_DIR / "artifacts" / "models" / "model.json"))
gain_b = booster_b.get_score(importance_type="gain")
sorted_b = sorted(gain_b.items(), key=lambda x: x[1], reverse=True)

# 2. Train Variant C (18 features with dynamic clock)
spec = FeatureSpec()
feat_c = spec.variant_c_features
X_train_c = train_df[feat_c].astype(np.float32)
y_train = train_df["is_error"].values
X_val_c = val_df[feat_c].astype(np.float32)
y_val = val_df["is_error"].values

clf_c = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    tree_method="hist",
    device="cpu",
)
clf_c.fit(X_train_c, y_train, eval_set=[(X_val_c, y_val)], verbose=False)
booster_c = clf_c.get_booster()
gain_c = booster_c.get_score(importance_type="gain")
sorted_c = sorted(gain_c.items(), key=lambda x: x[1], reverse=True)

# 3. Create Paired 2-Panel Figure
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

# Panel A: Variant B (16 features)
f_names_b = [x[0] for x in sorted_b][::-1]
f_scores_b = [x[1] for x in sorted_b][::-1]
y_pos_b = np.arange(len(f_names_b))

bars_b = ax1.barh(y_pos_b, f_scores_b, color="#2b5c8f", align="center", alpha=0.9)
ax1.set_yticks(y_pos_b)
ax1.set_yticklabels(f_names_b, fontsize=10)
ax1.set_xlabel("Average Gain per Split (XGBoost)", fontsize=11)
ax1.set_title("(a) Основная модель (Вариант Б, 16 признаков)", fontsize=12, fontweight="bold")
ax1.set_xlim(0, max(f_scores_b) * 1.08)

# Add value labels at end of bars
for bar in bars_b:
    w = bar.get_width()
    ax1.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", ha="left", fontsize=9, color="#222")

# Panel B: Variant C (18 features with clock)
f_names_c = [x[0] for x in sorted_c][::-1]
f_scores_c = [x[1] for x in sorted_c][::-1]
y_pos_c = np.arange(len(f_names_c))

# Highlight clock features
colors_c = ["#d95f02" if "clock" in name or "time" in name else "#2b5c8f" for name in f_names_c]
bars_c = ax2.barh(y_pos_c, f_scores_c, color=colors_c, align="center", alpha=0.9)
ax2.set_yticks(y_pos_c)
ax2.set_yticklabels(f_names_c, fontsize=10)
ax2.set_xlabel("Average Gain per Split (XGBoost)", fontsize=11)
ax2.set_title("(b) Расширенная модель (Вариант В, 18 признаков с часами)", fontsize=12, fontweight="bold")
ax2.set_xlim(0, max(f_scores_c) * 1.08)

for bar in bars_c:
    w = bar.get_width()
    ax2.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", ha="left", fontsize=9, color="#222")

# Add small legend for panel B colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2b5c8f", label="Позиционные и рейтинговые признаки"),
    Patch(facecolor="#d95f02", label="Временные признаки (часы и регламент)"),
]
ax2.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

plt.suptitle("Figure 3: Feature Importance Hierarchy by Average Gain (XGBoost)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save backup first
fig3_path = figures_dir / "figure3_feature_importance.png"
backup_path = figures_dir / "backup_figure3_feature_importance_1panel.png"
if fig3_path.exists() and not backup_path.exists():
    import shutil
    shutil.copyfile(fig3_path, backup_path)
    print(f"[Backup] Saved original Figure 3 to: {backup_path}")

fig.savefig(fig3_path)
plt.close(fig)
print(f"[SUCCESS] Paired Figure 3 saved to: {fig3_path}")
