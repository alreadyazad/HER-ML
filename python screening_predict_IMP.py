import pandas as pd
import numpy as np
import joblib

# ---------------- load screening alloys ----------------
df = pd.read_csv("screening_NiFeCo.csv")

elements = ['Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac','Mn_frac','Cr_frac','V_frac']
for c in elements:
    if c not in df.columns:
        df[c] = 0.0

# ---------------- atomic tables ----------------
EN_table = {'Ni':1.91,'Fe':1.83,'Co':1.88,'Mo':2.16,'Cu':1.90,'Mn':1.55,'Cr':1.66,'V':1.63}
R_table  = {'Ni':124,'Fe':132,'Co':125,'Mo':139,'Cu':128,'Mn':127,'Cr':128,'V':134}
VEC_table= {'Ni':10,'Fe':8,'Co':9,'Mo':6,'Cu':11,'Mn':7,'Cr':6,'V':5}
d_table  = {'Ni':8,'Fe':6,'Co':7,'Mo':5,'Cu':10,'Mn':5,'Cr':5,'V':3}

# ---------------- compute descriptors ----------------
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

    EN_var = np.dot(w, (EN - EN_mean)**2)
    R_var  = np.dot(w, (R - R_mean)**2)

    EN_std = np.sqrt(EN_var)
    R_std  = np.sqrt(R_var)

    EN_mis = np.max(np.abs(EN - EN_mean) * (fracs>0))
    R_mis  = np.max(np.abs(R - R_mean) * (fracs>0))

    VEC_mismatch = np.max(VEC) - np.min(VEC)

    return pd.Series({
        'EN_mean':EN_mean,
        'EN_std':EN_std,
        'EN_var':EN_var,
        'EN_mismatch':EN_mis,
        'Radius_mean':R_mean,
        'Radius_std':R_std,
        'Radius_var':R_var,
        'Radius_mismatch':R_mis,
        'VEC':VEC_mean,
        'VEC_mismatch':VEC_mismatch,
        'd_band_proxy':d_proxy
    })

df = pd.concat([df, df.apply(compute_feats, axis=1)], axis=1)

# ---------------- mixing entropy ----------------
def mix_entropy(row):
    f = np.array([row[c] for c in elements], float)
    f = f[f>0]
    return float(-np.sum(f*np.log(f))) if len(f)>0 else 0.0

df['mix_entropy'] = df.apply(mix_entropy, axis=1)

# ---------------- BUILD FEATURE MATRIX (NUMPY, EXACT ORDER, WITH DUPLICATES) ----------------
X = np.column_stack([
    df['Ni_frac'], df['Fe_frac'], df['Co_frac'], df['Mo_frac'],
    df['Cu_frac'], df['Mn_frac'], df['Cr_frac'], df['V_frac'],

    df['EN_mean'], df['EN_mean'],     # duplicate
    df['EN_std'], df['EN_var'], df['EN_mismatch'],

    df['Radius_mean'], df['Radius_mean'],   # duplicate
    df['Radius_std'], df['Radius_var'], df['Radius_mismatch'],

    df['VEC'], df['VEC'],             # duplicate
    df['VEC_mismatch'],

    df['mix_entropy'],
    df['d_band_proxy']
])

# ---------------- load trained full-feature model ----------------
model = joblib.load("rf_fullfeature_model.joblib")

# ---------------- predict ----------------
df['Predicted_DeltaG_H'] = model.predict(X)

# ---------------- shortlist good catalysts ----------------
good = df[(df['Predicted_DeltaG_H'] > -0.1) & (df['Predicted_DeltaG_H'] < 0.1)]

df.to_csv("screening_predictions_all.csv", index=False)
good.to_csv("screening_best_candidates.csv", index=False)

print("Total screened:", len(df))
print("Good candidates:", len(good))
print("DONE — screening successful.")
#----------------LOAD THE screening_NiFeCo AND COMPUTES THE FEATURES, NO ML----------