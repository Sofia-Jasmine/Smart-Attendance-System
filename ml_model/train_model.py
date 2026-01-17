import joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURES = [
    "hour",
    "attendance_frequency",
    "same_time_count",
    "subject_diversity",
    "day_variance",
    "is_fixed_time"
]

def train_model(df):
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, "models/proxy_attendance_rf.pkl")
    joblib.dump(FEATURES, "artifacts/model_features.pkl")

    print("✅ Model & features saved")
