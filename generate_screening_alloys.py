import pandas as pd
import numpy as np

compositions = []

step = 0.1  # 10% resolution

for ni in np.arange(0.4, 0.9, step):
    for fe in np.arange(0.0, 1.0, step):
        co = 1.0 - ni - fe
        if co < 0 or co > 1:
            continue

        compositions.append({
            'Ni_frac': round(ni, 2),
            'Fe_frac': round(fe, 2),
            'Co_frac': round(co, 2),
            'Mo_frac': 0.0,
            'Cu_frac': 0.0,
            'Mn_frac': 0.0,
            'Cr_frac': 0.0,
            'V_frac': 0.0
        })

df = pd.DataFrame(compositions)
df.to_csv("screening_NiFeCo.csv", index=False)

print("Generated", len(df), "Ni–Fe–Co screening alloys")
