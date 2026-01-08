import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ---------------- load data ----------------
df = pd.read_csv("her_dataset_200.csv")

elements = ['Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac','Mn_frac','Cr_frac','V_frac']
for c in elements:
    if c not in df.columns:
        df[c] = 0.0

EN_table = {'Ni':1.91,'Fe':1.83,'Co':1.88,'Mo':2.16,'Cu':1.90,'Mn':1.55,'Cr':1.66,'V':1.63}
R_table  = {'Ni':124,'Fe':132,'Co':125,'Mo':139,'Cu':128,'Mn':127,'Cr':128,'V':134}
VEC_table = {'Ni':10,'Fe':8,'Co':9,'Mo':6,'Cu':11,'Mn':7,'Cr':6,'V':5}
d_table = {'Ni':8,'Fe':6,'Co':7,'Mo':5,'Cu':10,'Mn':5,'Cr':5,'V':3}

def compute_feats(row):
    fracs = np.array([row[c] for c in elements], float)
    total = fracs.sum() if fracs.sum()!=0 else 1.0
    w = fracs / total
    elems = ['Ni','Fe','Co','Mo','Cu','Mn','Cr','V']

    EN = np.array([EN_table[e] for e in elems])
    R  = np.array([R_table[e] for e in elems])
    VEC= np.array([VEC_table[e] for e in elems])
    d  = np.array([d_table[e] for e in elems])

    EN_mean = np.dot(w, EN)
    R_mean  = np.dot(w, R)
    VEC_mean= np.dot(w, VEC)
    d_proxy = np.dot(w, d)

    EN_std = np.sqrt(np.dot(w, (EN - EN_mean)**2))
    R_std  = np.sqrt(np.dot(w, (R - R_mean)**2))

    EN_var = EN_std**2
    R_var  = R_std**2

    EN_mismatch = np.max(np.abs(EN - EN_mean) * (fracs > 0))
    R_mismatch  = np.max(np.abs(R - R_mean) * (fracs > 0))

    VEC_mismatch = np.max(VEC) - np.min(VEC)

    return pd.Series({
        'EN_mean':EN_mean,
        'EN_std':EN_std,
        'EN_var':EN_var,
        'EN_mismatch':EN_mismatch,
        'Radius_mean':R_mean,
        'Radius_std':R_std,
        'Radius_var':R_var,
        'Radius_mismatch':R_mismatch,
        'VEC':VEC_mean,
        'VEC_mismatch':VEC_mismatch,
        'd_band_proxy':d_proxy
    })

df = pd.concat([df, df.apply(compute_feats, axis=1)], axis=1)

def mix_entropy(row):
    f = np.array([row[c] for c in elements], float)
    f = f[f > 0]
    return -np.sum(f*np.log(f)) if len(f)>0 else 0.0

df['mix_entropy'] = df.apply(mix_entropy, axis=1)

features = elements + [
    'EN_mean','EN_std','EN_var','EN_mismatch',
    'Radius_mean','Radius_std','Radius_var','Radius_mismatch',
    'VEC','VEC_mismatch',
    'mix_entropy','d_band_proxy'
]

X = df[features]
y = df['DeltaG_H']

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)

pred = model.predict(Xte)

print("R2:", round(r2_score(yte,pred),3))
print("MAE:", round(mean_absolute_error(yte,pred),4))

joblib.dump(model, "rf_fullfeature_model_FIXED.joblib")
print("Saved rf_fullfeature_model_FIXED.joblib")