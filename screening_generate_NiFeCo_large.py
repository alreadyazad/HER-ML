import pandas as pd
import numpy as np

compositions = []

step = 0.010   # 2% step → ~1000+ alloys

for ni in np.arange(0.0, 1.0 + step, step):
    for fe in np.arange(0.0, 1.0 + step, step):
        co = 1.0 - ni - fe
        if co < 0 or co > 1:
            continue

        compositions.append({
            'Ni_frac': round(ni, 3),
            'Fe_frac': round(fe, 3),
            'Co_frac': round(co, 3),
            'Mo_frac': 0.0,
            'Cu_frac': 0.0,
            'Mn_frac': 0.0,
            'Cr_frac': 0.0,
            'V_frac': 0.0
        })

df = pd.DataFrame(compositions)
df.to_csv("screening_NiFeCo_large.csv", index=False)

print("Total alloys generated:", len(df))