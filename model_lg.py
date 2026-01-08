import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

model = LinearRegression()
model.fit(Xtr, ytr)

pred = model.predict(Xte)

print("LR R2:", round(r2_score(yte, pred), 3))
print("LR MAE:", round(mean_absolute_error(yte, pred), 4))

joblib.dump(model, "lr_model.joblib")
print("Saved lr_model.joblib")