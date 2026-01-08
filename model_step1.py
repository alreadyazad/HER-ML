import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load your dataset
df = pd.read_csv("her_dataset_200.csv")
print("Dataset loaded! Shape:", df.shape)

# 2. Choose the features (X) and target (y) WHAT FEAUTRES WE WANT AND FROM THAT FEAUTRES WHAT WE WANT TO CALCULATE
features = [
    'Ni_frac','Fe_frac','Co_frac','Mo_frac','Cu_frac',
    'Mn_frac','Cr_frac','V_frac','EN_mean','Radius_mean','VEC'
]
target = 'DeltaG_H'

X = df[features]
y = df[target]

# 3. Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the RANDON FOREST model
#Random Forest is chosen because it works well with small datasets.

#Uses many decision trees to learn non-linear patterns.
model = RandomForestRegressor(n_estimators=300, random_state=42)

# 5. Train it
model.fit(X_train, y_train)
print("Model trained!")

# 6. Test it
y_pred = model.predict(X_test)

# 7. Evaluate it
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# 8. Show feature importance
importances = model.feature_importances_
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.3f}")
