# model_step3_entropy.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --------------- load data ----------------
df = pd.read_csv("her_dataset_200.csv")

# element fraction columns we expect
elements = ['Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac','Mn_frac','Cr_frac','V_frac']
for c in elements:
    if c not in df.columns:
        df[c] = 0.0

# ---------------- atomic tables (proxies) ---------------
EN_table = {'Ni':1.91,'Fe':1.83,'Co':1.88,'Mo':2.16,'Cu':1.90,'Mn':1.55,'Cr':1.66,'V':1.63}
Radius_table = {'Ni':124,'Fe':132,'Co':125,'Mo':139,'Cu':128,'Mn':127,'Cr':128,'V':134}
VEC_table = {'Ni':10,'Fe':8,'Co':9,'Mo':6,'Cu':11,'Mn':7,'Cr':6,'V':5}

# --------------- compute / ensure descriptors exist ----------------
def compute_all(row):
    fracs = np.array([row[c] for c in elements], dtype=float)
    elems = ['Ni','Fe','Co','Mo','Cu','Mn','Cr','V']
    en_vals = np.array([EN_table[e] for e in elems], dtype=float)
    r_vals = np.array([Radius_table[e] for e in elems], dtype=float)
    vec_vals = np.array([VEC_table[e] for e in elems], dtype=float)

    total = fracs.sum()
    if total == 0:
        total = 1.0
    w = fracs / total

    en_mean = float(np.dot(w, en_vals))
    r_mean = float(np.dot(w, r_vals))
    vec_mean = float(np.dot(w, vec_vals))

    en_var = float(np.dot(w, (en_vals - en_mean)**2))
    r_var = float(np.dot(w, (r_vals - r_mean)**2))
    en_std = float(np.sqrt(en_var))
    r_std = float(np.sqrt(r_var))

    # mismatch = max |value-mean| among present elements
    if (fracs > 0).any():
        en_mismatch = float(np.max(np.abs(en_vals - en_mean) * (fracs > 0)))
        r_mismatch = float(np.max(np.abs(r_vals - r_mean) * (fracs > 0)))
    else:
        en_mismatch = 0.0
        r_mismatch = 0.0

    return pd.Series({
        'EN_mean': en_mean,
        'EN_std': en_std,
        'EN_mismatch': en_mismatch,
        'Radius_mean': r_mean,
        'Radius_std': r_std,
        'Radius_mismatch': r_mismatch,
        'VEC': vec_mean
    })

# Only compute if missing (but safe to recompute)
computed = df.apply(compute_all, axis=1)
for col in computed.columns:
    df[col] = computed[col]

# ---------------- mixing entropy ----------------
def mixing_entropy(row):
    fracs = np.array([row[c] for c in elements], dtype=float)
    fracs = fracs[fracs > 0]
    if len(fracs) == 0:
        return 0.0
    return float(-np.sum(fracs * np.log(fracs)))

df['mix_entropy'] = df.apply(mixing_entropy, axis=1)

# ---------------- prepare features and target ----------------
features = elements + [
    'EN_mean','EN_std','EN_mismatch',
    'Radius_mean','Radius_std','Radius_mismatch',
    'VEC','mix_entropy'
]
target = 'DeltaG_H'

# drop rows missing target
df = df.dropna(subset=[target]).reset_index(drop=True)

X = df[features].values
y = df[target].values

print("Prepared X shape:", X.shape, "y shape:", y.shape)
print("Using features:", features)

# ---------------- train/test and model ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
