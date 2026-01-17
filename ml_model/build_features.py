import pandas as pd

def build_features(csv_path):
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(csv_path)

    # -----------------------------
    # STANDARDIZE COLUMN NAMES
    # Roll_Number -> roll_number
    # Date        -> date
    # Time        -> time
    # Subject     -> subject
    # Status      -> status
    # Label       -> label
    # -----------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # -----------------------------
    # VALIDATE REQUIRED COLUMNS
    # -----------------------------
    required_cols = {
        "roll_number",
        "date",
        "time",
        "subject",
        "status",
        "label"
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # -----------------------------
    # DATE & TIME PROCESSING
    # -----------------------------
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M").dt.hour

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    # Total attendance count per student
    df["attendance_frequency"] = (
        df.groupby("roll_number")["date"].transform("count")
    )

    # Repeated attendance at same exact time
    df["same_time_count"] = (
        df.groupby(["roll_number", "time"])["time"].transform("count")
    )

    # Number of unique subjects attended
    df["subject_diversity"] = (
        df.groupby("roll_number")["subject"].transform("nunique")
    )

    # Number of unique weekdays attended
    df["day_variance"] = (
        df.groupby("roll_number")["date"]
        .transform(lambda x: x.dt.dayofweek.nunique())
    )

    # Fixed-time attendance flag
    df["is_fixed_time"] = (df["same_time_count"] >= 5).astype(int)

    return df
