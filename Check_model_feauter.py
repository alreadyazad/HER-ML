import joblib

model = joblib.load("rf_fullfeature_model.joblib")
print(list(model.feature_names_in_))