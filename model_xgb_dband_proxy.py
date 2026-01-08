import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("dband_proxy_dataset_large.csv")

features = [
    'Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac',
    'd_band_proxy'
]

X = df[features].values
y = df['DeltaG_H_eV'].values   # 🔥 correct column

# =====================
# TRAIN TEST SPLIT
# =====================
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# MODEL
# =====================
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
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
mae_pct = np.mean(np.abs(pred - yte) / np.maximum(np.abs(yte), 1e-6)) * 100

print("d-band PROXY MODEL")
print("R2:", round(r2_score(yte, pred), 3))
print("MAE (eV):", round(mae_ev, 4))
print("MAE (%):", round(mae_pct, 2))