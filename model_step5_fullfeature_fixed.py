import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_csv("her_dataset_200.csv")

elements = ['Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac','Mn_frac','Cr_frac','V_frac']
for c in elements:
    if c not in df.columns:
        df[c] = 0.0

EN = {'Ni':1.91,'Fe':1.83,'Co':1.88,'Mo':2.16,'Cu':1.90,'Mn':1.55,'Cr':1.66,'V':1.63}
R  = {'Ni':124,'Fe':132,'Co':125,'Mo':139,'Cu':128,'Mn':127,'Cr':128,'V':134}
VEC= {'Ni':10,'Fe':8,'Co':9,'Mo':6,'Cu':11,'Mn':7,'Cr':6,'V':5}
D  = {'Ni':8,'Fe':6,'Co':7,'Mo':5,'Cu':10,'Mn':5,'Cr':5,'V':3}

def compute(row):
    f = np.array([row[e] for e in elements], float)
    w = f / (f.sum() if f.sum()!=0 else 1)
    elems = ['Ni','Fe','Co','Mo','Cu','Mn','Cr','V']

    ENv = np.array([EN[e] for e in elems])
    Rv  = np.array([R[e] for e in elems])
    Vv  = np.array([VEC[e] for e in elems])
    Dv  = np.array([D[e] for e in elems])

    return pd.Series({
        'EN_mean': np.dot(w,ENv),
        'EN_std': np.sqrt(np.dot(w,(ENv-np.dot(w,ENv))**2)),
        'EN_mismatch': np.max(np.abs(ENv-np.dot(w,ENv))*(f>0)),
        'Radius_mean': np.dot(w,Rv),
        'Radius_std': np.sqrt(np.dot(w,(Rv-np.dot(w,Rv))**2)),
        'Radius_mismatch': np.max(np.abs(Rv-np.dot(w,Rv))*(f>0)),
        'VEC': np.dot(w,Vv),
        'mix_entropy': -np.sum(f[f>0]*np.log(f[f>0])) if (f>0).any() else 0,
        'd_band_proxy': np.dot(w,Dv)
    })

df = pd.concat([df, df.apply(compute, axis=1)], axis=1)

features = elements + [
    'EN_mean','EN_std','EN_mismatch',
    'Radius_mean','Radius_std','Radius_mismatch',
    'VEC','mix_entropy','d_band_proxy'
]

X = df[features]
y = df['DeltaG_H']

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(Xtr,ytr)

print("R2:", r2_score(yte, model.predict(Xte)))
print("MAE:", mean_absolute_error(yte, model.predict(Xte)))

joblib.dump(model,"rf_fullfeature_model_fixed.joblib")
print("Saved rf_fullfeature_model_fixed.joblib")