# model_step2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import time

# 1) Load dataset
df = pd.read_csv("her_dataset_200.csv")
print("Loaded dataset:", df.shape)

# 2) Create extra simple descriptors
# We assume columns like Ni_frac, Fe_frac, ... V_frac, EN_mean, Radius_mean, VEC exist
element_cols = ['Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac','Mn_frac','Cr_frac','V_frac']

# safety: fill missing element columns with 0 if any not present
for c in element_cols:
    if c not in df.columns:
        df[c] = 0.0

# Basic descriptor engineering
# (a) weighted std of electronegativity approximated by composition and element EN values
# For simplicity we will use EN values for common elements (dictionary)
EN_table = {'Ni':1.91,'Fe':1.83,'Co':1.88,'Mo':2.16,'Cu':1.90,'Mn':1.55,'Cr':1.66,'V':1.63}
Radius_table = {'Ni':124,'Fe':132,'Co':125,'Mo':139,'Cu':128,'Mn':127,'Cr':128,'V':134}

# compute weighted std and max diff proxies
def compute_agg(row):
    # build arrays of present elements and fractions
    fracs = np.array([row[c] for c in element_cols], dtype=float)
    elems = ['Ni','Fe','Co','Mo','Cu','Mn','Cr','V']
    en_vals = np.array([EN_table[e] for e in elems], dtype=float)
    r_vals = np.array([Radius_table[e] for e in elems], dtype=float)

    # weighted mean already present (EN_mean, Radius_mean) but compute them to be safe
    total = fracs.sum()
    if total == 0:
        total = 1.0
    weights = fracs/total
    en_mean_w = np.dot(weights, en_vals)
    r_mean_w = np.dot(weights, r_vals)

    # weighted variance/std
    en_var = np.dot(weights, (en_vals - en_mean_w)**2)
    r_var = np.dot(weights, (r_vals - r_mean_w)**2)
    en_std = np.sqrt(en_var)
    r_std = np.sqrt(r_var)

    # mismatch = max absolute difference between any element EN and mean
    en_mismatch = np.max(np.abs(en_vals - en_mean_w) * (fracs>0)) if (fracs>0).any() else 0.0
    r_mismatch = np.max(np.abs(r_vals - r_mean_w) * (fracs>0)) if (fracs>0).any() else 0.0

    return en_mean_w, r_mean_w, en_std, r_std, en_mismatch, r_mismatch

computed = df.apply(lambda r: pd.Series(compute_agg(r), index=['EN_mean_calc','Radius_mean_calc','EN_std','Radius_std','EN_mismatch','Radius_mismatch']), axis=1)
# merge - prefer existing EN_mean/Radius_mean if present, otherwise use calc
for col in ['EN_mean','Radius_mean']:
    if col not in df.columns or df[col].isnull().all():
        df[col] = computed[col + '_calc']
# attach new cols
df = pd.concat([df, computed[['EN_std','Radius_std','EN_mismatch','Radius_mismatch']]], axis=1)

# 3) Features and target
features = element_cols + ['EN_mean','EN_std','EN_mismatch','Radius_mean','Radius_std','Radius_mismatch','VEC']
target = 'DeltaG_H'

# Drop rows with missing target
df = df.dropna(subset=[target])

X = df[features].values
y = df[target].values

print("Using features:", features)
print("X shape:", X.shape, "y shape:", y.shape)

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# 5) Baseline cross-validated score with default RF
base_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(base_rf, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print("Baseline CV R² (5-fold) mean:", np.mean(cv_scores).round(3), "std:", np.std(cv_scores).round(3))

# 6) Hyperparameter tuning with RandomizedSearchCV (fast)
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 8, 12, 16, 24],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 0.5, 0.8]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=30, cv=4, scoring='r2', random_state=42, n_jobs=-1, verbose=1)

t0 = time.time()
rand_search.fit(X_train, y_train)
t1 = time.time()
print(f"RandomizedSearchCV done in {int(t1-t0)} sec. Best score (cv):", rand_search.best_score_)
print("Best params:", rand_search.best_params_)

best_model = rand_search.best_estimator_

# 7) Evaluate on test set
y_pred_test = best_model.predict(X_test)
print("Test R²:", round(r2_score(y_test, y_pred_test), 3))
print("Test MAE:", round(mean_absolute_error(y_test, y_pred_test), 4))


# 8) Cross-validated R² on full data for reporting
cv_full = cross_val_score(best_model, X, y, cv=5, scoring='r2', n_jobs=-1)
print("Final model 5-fold CV R² mean:", np.mean(cv_full).round(3), "std:", np.std(cv_full).round(3))

# 9) Feature importance (permutation importance for robust view)
result = permutation_importance(best_model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
importances = result.importances_mean
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{feat}: {imp:.4f}")

# 10) Save model
joblib.dump(best_model, "rf_best_model_step2.joblib")
print("Saved best model to rf_best_model_step2.joblib")
