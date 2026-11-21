import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import xgboost as xgb
from chess_error_prediction.features.feature_spec import FeatureSpec

train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/val.csv")

# 1. Variant B (16 features) from canonical model.json
booster_b = xgb.Booster()
booster_b.load_model("artifacts/models/model.json")
gain_b = booster_b.get_score(importance_type="gain")
sorted_b = sorted(gain_b.items(), key=lambda x: x[1], reverse=True)

# 2. Variant C (18 features: Variant B + clock_remaining + time_spent_prev_move)
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

print("=" * 86)
print("ТОЧНЫЕ ЗНАЧЕНИЯ ВАЖНОСТИ ПРИЗНАКОВ (GAIN) ДЛЯ ОБЕИХ МОДЕЛЕЙ")
print("=" * 86)
h_rank = "Ранг"
h_b = "Вариант Б (16 признаков, основная)"
h_c = "Вариант В (18 признаков, с часами)"
print(f"{h_rank:<5} | {h_b:<36} | {h_c:<36}")
print("-" * 86)

max_len = max(len(sorted_b), len(sorted_c))
for i in range(max_len):
    b_str = f"{sorted_b[i][0]}: {sorted_b[i][1]:.2f}" if i < len(sorted_b) else "-"
    c_str = f"{sorted_c[i][0]}: {sorted_c[i][1]:.2f}" if i < len(sorted_c) else "-"
    print(f"{i+1:<5} | {b_str:<36} | {c_str:<36}")

print("=" * 86)
print(f"Параметры обучения: n_estimators=300, max_depth=4, lr=0.05, early_stopping=30, seed=42, CPU")
print(f"Лучшая итерация (best_iteration): Вариант Б = 149, Вариант В = {clf_c.best_iteration}")
