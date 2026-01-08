import pandas as pd
import numpy as np

compositions = []

step = 0.010   # 2% step → ~1000+ alloys

for ni in np.arange(0.0, 1.0 + step, step):
    for mo in np.arange(0.0, 1.0 + step, step):
        cu = 1.0 - ni - mo
        if cu < 0 or cu > 1:
            continue

        compositions.append({
            'Ni_frac': round(ni, 3),
            'Fe_frac': 0.0,
            'Co_frac': 0.0,
            'Mo_frac': round(mo,3),
            'Cu_frac': round(cu,3),
            'Mn_frac': 0.0,
            'Cr_frac': 0.0,
            'V_frac': 0.0
        })

df = pd.DataFrame(compositions)
df.to_csv("screening_NiMoCu_large.csv", index=False)

print("Total alloys generated:", len(df))