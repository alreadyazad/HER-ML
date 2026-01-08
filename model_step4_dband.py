# model_step4_dband.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("her_dataset_200.csv")

elements = ['Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac','Mn_frac','Cr_frac','V_frac']
for c in elements:
    if c not in df.columns:
        df[c] = 0.0

# ---- d-electron table ----
d_table = {
    'Ni': 8, 'Fe': 6, 'Co': 7, 'Mo': 5,
    'Cu':10, 'Mn':5, 'Cr':5, 'V':3
}

# ---- other atomic tables reused ----
EN_table = {'Ni':1.91,'Fe':1.83,'Co':1.88,'Mo':2.16,'Cu':1.90,'Mn':1.55,'Cr':1.66,'V':1.63}
Radius_table = {'Ni':124,'Fe':132,'Co':125,'Mo':139,'Cu':128,'Mn':127,'Cr':128,'V':134}
VEC_table = {'Ni':10,'Fe':8,'Co':9,'Mo':6,'Cu':11,'Mn':7,'Cr':6,'V':5}

# ---- compute descriptors ----
def compute_all(row):
    fracs = np.array([row[c] for c in elements], float)
    elems = ['Ni','Fe','Co','Mo','Cu','Mn','Cr','V']

    w = fracs / (fracs.sum() if fracs.sum() != 0 else 1)

    en_vals = np.array([EN_table[e] for e in elems])
    r_vals = np.array([Radius_table[e] for e in elems])
    d_vals = np.array([d_table[e] for e in elems])
    vec_vals = np.array([VEC_table[e] for e in elems])

    EN_mean = float(np.dot(w, en_vals))
    Radius_mean = float(np.dot(w, r_vals))
    VEC = float(np.dot(w, vec_vals))
    d_band_proxy = float(np.dot(w, d_vals))

    EN_std = float(np.sqrt(np.dot(w, (en_vals - EN_mean)**2)))
    Radius_std = float(np.sqrt(np.dot(w, (r_vals - Radius_mean)**2)))

    EN_mismatch = float(np.max(np.abs(en_vals - EN_mean) * (fracs > 0)))
    Radius_mismatch = float(np.max(np.abs(r_vals - Radius_mean) * (fracs > 0)))

    return pd.Series({
        'EN_mean': EN_mean,
        'EN_std': EN_std,
        'EN_mismatch': EN_mismatch,
        'Radius_mean': Radius_mean,
        'Radius_std': Radius_std,
        'Radius_mismatch': Radius_mismatch,
        'VEC': VEC,
        'd_band_proxy': d_band_proxy
    })

computed = df.apply(compute_all, axis=1)
df = pd.concat([df, computed], axis=1)

# ---- mixing entropy ----
def mixing_entropy(row):
    f = np.array([row[c] for c in elements], float)
    f = f[f>0]
    return float(-np.sum(f * np.log(f)))

df['mix_entropy'] = df.apply(mixing_entropy, axis=1)

# ---- build features ----
features = elements + [
    'EN_mean','EN_std','EN_mismatch',
    'Radius_mean','Radius_std','Radius_mismatch',
    'VEC','mix_entropy','d_band_proxy'
]

X = df[features].values
y = df['DeltaG_H'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("Done.")
