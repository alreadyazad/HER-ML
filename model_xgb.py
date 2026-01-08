import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("her_dataset_200.csv")

# =====================
# SELECT FEATURES
# =====================
features = [
    'Ni_frac','Fe_frac','Co_frac','Mo_frac',
    'Cu_frac','Mn_frac','Cr_frac','V_frac',
    'EN_mean','Radius_mean','VEC'
]

# Keep only required columns
df = df[features + ['DeltaG_H']]

# =====================
# FORCE NUMERIC (IMPORTANT)
# =====================
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

X = df[features].values   # 🔥 convert to NumPy
y = df['DeltaG_H'].values

# =====================
# TRAIN–TEST SPLIT
# =====================
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# XGBOOST MODEL
# =====================
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(Xtr, ytr)

# =====================
# EVALUATION
# =====================
y_pred = model.predict(Xte)

print("XGB R2:", round(r2_score(yte, y_pred), 3))
print("XGB MAE:", round(mean_absolute_error(yte, y_pred), 4))

# =====================
# SAVE MODEL
# =====================
joblib.dump(model, "xgb_model_fixed.joblib")
print("Saved xgb_model_fixed.joblib")

# =====================
# TRUE vs PREDICTED PLOT
# =====================
plt.figure(figsize=(5,5))
plt.scatter(yte, y_pred, s=30)
plt.plot(
    [yte.min(), yte.max()],
    [yte.min(), yte.max()],
    'k--', lw=2
)
plt.xlabel("True ΔG_H (eV)")
plt.ylabel("Predicted ΔG_H (eV)")
plt.title("XGBoost: True vs Predicted")
plt.tight_layout()
plt.savefig("true_vs_pred_xgb.png", dpi=300)
plt.show()