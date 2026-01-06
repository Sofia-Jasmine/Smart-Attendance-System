import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# =========================
# PATH SETUP (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "attendance_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# STANDARDIZE COLUMN NAMES
# =========================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Expected columns:
# roll_number, date, time, subject, status, label

# =========================
# PREPROCESSING
# =========================
df["date"] = pd.to_datetime(df["date"])
df["status"] = df["status"].map({"Present": 1, "Absent": 0})

df["hour"] = pd.to_datetime(df["time"], format="%H:%M").dt.hour
df["minute"] = pd.to_datetime(df["time"], format="%H:%M").dt.minute

# =========================
# FEATURE ENGINEERING
# =========================
df["attendance_frequency"] = df.groupby("roll_number")["date"].transform("count")
df["same_time_count"] = df.groupby(["roll_number", "time"])["time"].transform("count")
df["subject_diversity"] = df.groupby("roll_number")["subject"].transform("nunique")
df["day_variance"] = df.groupby("roll_number")["date"].transform(
    lambda x: x.dt.dayofweek.nunique()
)

# =========================
# SELECT FEATURES & LABEL
# =========================
FEATURES = [
    "status",
    "hour",
    "minute",
    "attendance_frequency",
    "same_time_count",
    "subject_diversity",
    "day_variance"
]

X = df[FEATURES]
y = df["label"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_score = cross_val_score(model, X, y, cv=5).mean()

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Mean CV Accuracy: {cv_score * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# CONFIDENCE SCORES
# =========================
y_proba = model.predict_proba(X_test)
confidence_scores = np.max(y_proba, axis=1)

confidence_df = pd.DataFrame({
    "Predicted_Label": y_pred,
    "Confidence_Score": confidence_scores
})

confidence_df.to_csv(
    os.path.join(ARTIFACT_DIR, "predictions_with_confidence.csv"),
    index=False
)

# =========================
# SAVE MODEL & METADATA
# =========================
joblib.dump(model, os.path.join(MODEL_DIR, "proxy_attendance_rf.pkl"))
joblib.dump(FEATURES, os.path.join(ARTIFACT_DIR, "model_features.pkl"))

print("\n Model saved successfully:")
print(" - ml_model/models/proxy_attendance_rf.pkl")
print(" - ml_model/artifacts/model_features.pkl")
print(" - ml_model/artifacts/predictions_with_confidence.csv")
