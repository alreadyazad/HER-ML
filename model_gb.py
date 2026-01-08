import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_csv("her_dataset_200.csv")

features = [
    'Ni_frac','Fe_frac','Co_frac','Mo_frac',
    'Cu_frac','Mn_frac','Cr_frac','V_frac',
    'EN_mean','Radius_mean','VEC'
]

X = df[features].values
y = df['DeltaG_H'].values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(Xtr, ytr)

pred = model.predict(Xte)

print("GB R2:", round(r2_score(yte, pred), 3))
print("GB MAE:", round(mean_absolute_error(yte, pred), 4))

joblib.dump(model, "gb_model.joblib")
print("Saved gb_model.joblib")