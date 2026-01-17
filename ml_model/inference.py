import joblib
import numpy as np
import os

def ml_predict(feature_dict):
    model_path = "models/proxy_attendance_rf.pkl"
    features_path = "artifacts/model_features.pkl"

    model = joblib.load(model_path)
    features_order = joblib.load(features_path)

    X = np.array([[feature_dict[f] for f in features_order]])
    return model.predict_proba(X)[0][1]
