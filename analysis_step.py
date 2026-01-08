# analysis_step.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("her_dataset_200.csv")

# Base element fractions
element_cols = ['Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac','Mn_frac','Cr_frac','V_frac']
for c in element_cols:
    if c not in df.columns:
        df[c] = 0.0

# 2. ONLY compute missing descriptors: EN_std, EN_mismatch, Radius_std, Radius_mismatch
EN_table = {'Ni':1.91,'Fe':1.83,'Co':1.88,'Mo':2.16,'Cu':1.90,'Mn':1.55,'Cr':1.66,'V':1.63}
Radius_table = {'Ni':124,'Fe':132,'Co':125,'Mo':139,'Cu':128,'Mn':127,'Cr':128,'V':134}

def compute_extra(row):
    fracs = np.array([row[c] for c in element_cols])
    elems = ['Ni','Fe','Co','Mo','Cu','Mn','Cr','V']

    en_vals = np.array([EN_table[e] for e in elems])
    r_vals = np.array([Radius_table[e] for e in elems])

    total = fracs.sum() if fracs.sum() != 0 else 1.0
    w = fracs / total

    # Use EXISTING EN_mean and Radius_mean (DON’T recompute)
    en_mean = row['EN_mean']
    r_mean = row['Radius_mean']

    en_std = np.sqrt((w * (en_vals - en_mean)**2).sum())
    r_std = np.sqrt((w * (r_vals - r_mean)**2).sum())

    en_mis = np.max(np.abs(en_vals - en_mean) * (fracs > 0))
    r_mis = np.max(np.abs(r_vals - r_mean) * (fracs > 0))

    return pd.Series([en_std, en_mis, r_std, r_mis],
                     index=['EN_std','EN_mismatch','Radius_std','Radius_mismatch'])

computed = df.apply(compute_extra, axis=1)
df = pd.concat([df, computed], axis=1)

df = df.dropna(subset=['DeltaG_H']).reset_index(drop=True)

# 3. Load trained model
model = joblib.load("rf_best_model_step2.joblib")

features = element_cols + [
    'EN_mean','EN_std','EN_mismatch',
    'Radius_mean','Radius_std','Radius_mismatch',
    'VEC'
]

X = df[features].values
y = df['DeltaG_H'].values

# 4. Same split as training
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index.values, test_size=0.2, random_state=42)

# 5. Predict
y_pred = model.predict(X_test)
print("Test R²:", round(r2_score(y_test, y_pred), 3))
print("Test MAE:", round(mean_absolute_error(y_test, y_pred), 4))

# 6. Save predictions
out = pd.DataFrame(X_test, columns=features)
out['y_true'] = y_test
out['y_pred'] = y_pred
out['error'] = out['y_pred'] - out['y_true']
out['abs_error'] = abs(out['error'])
out.to_csv("predictions_test_set.csv", index=False)
print("Saved predictions_test_set.csv")

# 7. Plots
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel("True ΔG_H")
plt.ylabel("Predicted ΔG_H")
plt.title("True vs Predicted")
plt.savefig("true_vs_pred.png")
print("Saved true_vs_pred.png")

plt.figure(figsize=(6,4))
sns.histplot(out['error'], bins=20, kde=True)
plt.xlabel("Prediction Error")
plt.title("Residuals Histogram")
plt.savefig("residuals_hist.png")
print("Saved residuals_hist.png")

# 8. Correlation heatmap
corr = df[features + ['DeltaG_H']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig("feature_corr_heatmap.png")
print("Saved feature_corr_heatmap.png")

print("All done.")
