import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("screening_predictions_all.csv")

# -------- 1. ΔG_H* vs index --------
plt.figure()
plt.scatter(range(len(df)), df['Predicted_DeltaG_H'])
plt.axhline(0.1, linestyle='--')
plt.axhline(-0.1, linestyle='--')
plt.xlabel("Screened Alloy Index")
plt.ylabel("Predicted ΔG_H* (eV)")
plt.title("Screening Results")
plt.savefig("screening_deltaG_vs_index.png")

# -------- 2. Histogram --------
plt.figure()
plt.hist(df['Predicted_DeltaG_H'], bins=15)
plt.xlabel("Predicted ΔG_H* (eV)")
plt.ylabel("Count")
plt.title("Distribution of Predicted ΔG_H*")
plt.savefig("screening_histogram.png")

# -------- 3. Ni fraction vs ΔG_H* --------
plt.figure()
plt.scatter(df['Ni_frac'], df['Predicted_DeltaG_H'])
plt.xlabel("Ni Fraction")
plt.ylabel("Predicted ΔG_H* (eV)")
plt.title("Effect of Ni Content")
plt.savefig("Ni_vs_deltaG.png")

print("Plots saved.")