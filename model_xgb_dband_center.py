import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("dband_center_dataset_large.csv")

# ---------------------
# ONE-HOT ENCODE ELEMENT
# ---------------------
df = pd.get_dummies(df, columns=["material"], drop_first=False)

# ---------------------
# FEATURES
# ---------------------
features = ['d_band_center_eV'] + [c for c in df.columns if c.startswith("material_")]

X = df[features].values
y = df['DeltaG_H_eV'].values

# =====================
# TRAIN TEST SPLIT
# =====================
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =====================
# MODEL
# =====================
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

model.fit(Xtr, ytr)

pred = model.predict(Xte)

# =====================
# METRICS
# =====================
mae_ev = mean_absolute_error(yte, pred)

mae_pct = np.mean(
    np.abs(pred - yte) / np.maximum(np.abs(yte), 1e-6)
) * 100

print("d-band CENTER MODEL (DFT)")
print("R2:", round(r2_score(yte, pred), 3))
print("MAE (eV):", round(mae_ev, 4))
print("MAE (%):", round(mae_pct, 2))