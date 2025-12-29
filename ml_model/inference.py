import joblib
import pandas as pd

def flag_risk(prediction, confidence):
    if prediction == 1 and confidence >= 0.85:
        return "HIGH RISK (Proxy)"
    elif prediction == 1 and confidence >= 0.65:
        return "MEDIUM RISK"
    else:
        return "LOW RISK / NORMAL"


model = joblib.load("models/proxy_attendance_rf.pkl")
features = joblib.load("artifacts/model_features.pkl")

def predict_proxy(input_data: dict):
    X = pd.DataFrame([input_data], columns=features)

    prediction = model.predict(X)[0]
    confidence = float(model.predict_proba(X)[0].max())

    risk = flag_risk(prediction, confidence)

    return {
        "prediction": "Proxy" if prediction == 1 else "Normal",
        "confidence": round(confidence, 2),
        "risk_level": risk
    }

# TEST
if __name__ == "__main__":
    sample_input = {
        "status": 1,
        "hour": 9,
        "minute": 1,
        "attendance_frequency": 25,
        "same_time_count": 7,
        "subject_diversity": 1,
        "day_variance": 5
    }

    print(predict_proxy(sample_input))
