import pandas as pd
import numpy as np
import joblib

# ===============================
# 1. LOAD SCREENING COMPOSITIONS
# ===============================
df = pd.read_csv("screening_NiMoCu_large.csv")

elements = ['Ni_frac','Fe_frac','Co_frac','Mo_frac',
            'Cu_frac','Mn_frac','Cr_frac','V_frac']

for c in elements:
    if c not in df.columns:
        df[c] = 0.0

# ===============================
# 2. ATOMIC TABLES
# ===============================
EN = {'Ni':1.91,'Fe':1.83,'Co':1.88,'Mo':2.16,
      'Cu':1.90,'Mn':1.55,'Cr':1.66,'V':1.63}

R  = {'Ni':124,'Fe':132,'Co':125,'Mo':139,
      'Cu':128,'Mn':127,'Cr':128,'V':134}

VEC = {'Ni':10,'Fe':8,'Co':9,'Mo':6,
       'Cu':11,'Mn':7,'Cr':6,'V':5}

d_band = {'Ni':8,'Fe':6,'Co':7,'Mo':5,
          'Cu':10,'Mn':5,'Cr':5,'V':3}

elems = ['Ni','Fe','Co','Mo','Cu','Mn','Cr','V']

ENv = np.array([EN[e] for e in elems])
Rv  = np.array([R[e] for e in elems])
Vv  = np.array([VEC[e] for e in elems])
dv  = np.array([d_band[e] for e in elems])

# ===============================
# 3. FEATURE COMPUTATION (FIXED)
# ===============================
def compute_features(row):
    fracs = np.array([row[e] for e in elements], dtype=float)

    total = fracs.sum()
    if total == 0:
        return pd.Series(np.nan)

    w = fracs / total
    mask = fracs > 0

    EN_mean = np.dot(w, ENv)
    R_mean  = np.dot(w, Rv)
    VEC_mean = np.dot(w, Vv)
    d_proxy = np.dot(w, dv)

    EN_std = np.sqrt(np.dot(w, (ENv - EN_mean)**2))
    R_std  = np.sqrt(np.dot(w, (Rv - R_mean)**2))

    EN_var = EN_std**2
    R_var  = R_std**2

    EN_mis = np.max(np.abs(ENv[mask] - EN_mean))
    R_mis  = np.max(np.abs(Rv[mask] - R_mean))
    VEC_mis = np.max(Vv[mask]) - np.min(Vv[mask])

    fracs_safe = np.clip(fracs, 1e-12, 1.0)
    mix_entropy = -np.sum(fracs_safe * np.log(fracs_safe))

    return pd.Series({
        'EN_mean': EN_mean,
        'EN_std': EN_std,
        'EN_var': EN_var,
        'EN_mismatch': EN_mis,
        'Radius_mean': R_mean,
        'Radius_std': R_std,
        'Radius_var': R_var,
        'Radius_mismatch': R_mis,
        'VEC': VEC_mean,
        'VEC_mismatch': VEC_mis,
        'mix_entropy': mix_entropy,
        'd_band_proxy': d_proxy
    })

df_feat = df.apply(compute_features, axis=1)
df = pd.concat([df, df_feat], axis=1)
df = df.dropna()

# ===============================
# 4. LOAD MODEL
# ===============================
model = joblib.load("rf_fullfeature_model_FIXED.joblib")

features = model.feature_names_in_
X = df[features]

# ===============================
# 5. PREDICT ΔG_H
# ===============================
df['Predicted_DeltaG_H'] = model.predict(X)

# ===============================
# 6. FILTER GOOD CANDIDATES
# ===============================
best = df[
    (df['Predicted_DeltaG_H'] > -0.02) &
    (df['Predicted_DeltaG_H'] <  0.02)
]

# ===============================
# 7. SAVE RESULTS
# ===============================
df.to_csv("screening_predictions_large.csv", index=False)
best.to_csv("screening_best_candidates_large_NiMoCu.csv", index=False)

print("DONE.")
print("Total screened:", len(df))
print("Good candidates:", len(best))