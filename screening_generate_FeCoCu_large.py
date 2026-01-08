import pandas as pd
import numpy as np

compositions = []

step = 0.0125   # smaller step = more alloys

for fe in np.arange(0.0, 1.0 + step, step):
    for co in np.arange(0.0, 1.0 + step, step):
        cu = 1.0 - fe - co
        if cu < 0 or cu > 1:
            continue

        compositions.append({
            'Ni_frac': 0.0,
            'Fe_frac': round(fe, 3),
            'Co_frac': round(co, 3),
            'Cu_frac': round(cu, 3),
            'Mo_frac': 0.0,
            'Mn_frac': 0.0,
            'Cr_frac': 0.0,
            'V_frac': 0.0
        })

df = pd.DataFrame(compositions)
df.to_csv("screening_FeCoCu_large.csv", index=False)

print("Generated", len(df), "Fe–Co–Cu screening alloys")