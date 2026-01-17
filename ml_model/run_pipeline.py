from feature_engineering.build_features import build_features
from ml.train_model import train_model
from ml.inference import ml_predict
from logic.proxy_rules import apply_rules
import os

df = build_features("data/attendance_data.csv")
print("✅ Features built")

if not os.path.exists("models/proxy_attendance_rf.pkl"):
    train_model(df)

sample = df.iloc[0]

feature_input = {
    "hour": sample["hour"],
    "attendance_frequency": sample["attendance_frequency"],
    "same_time_count": sample["same_time_count"],
    "subject_diversity": sample["subject_diversity"],
    "day_variance": sample["day_variance"],
    "is_fixed_time": sample["is_fixed_time"]
}

probability = ml_predict(feature_input)
is_proxy, reasons = apply_rules(probability, feature_input)

print("\n🔍 FINAL DECISION")
print("----------------")
print("Proxy:", "YES" if is_proxy else "NO")
print("Confidence:", round(probability, 2))
print("Reasons:", reasons if reasons else "Normal behavior")
