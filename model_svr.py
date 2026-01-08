import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("her_dataset_200.csv")

features = [
    'Ni_frac','Fe_frac','Co_frac','Mo_frac',
    'Cu_frac','Mn_frac','Cr_frac','V_frac',
    'EN_mean','Radius_mean','VEC'
]

X = df[features].values
y = df['DeltaG_H'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVR(C=10, gamma='scale', epsilon=0.01)

model.fit(Xtr, ytr)

pred = model.predict(Xte)

print("SVR R2:", round(r2_score(yte, pred), 3))
print("SVR MAE:", round(mean_absolute_error(yte, pred), 4))

joblib.dump((model, scaler), "svr_model.joblib")
print("Saved svr_model.joblib")