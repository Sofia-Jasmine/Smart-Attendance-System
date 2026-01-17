📊 Smart Attendance System
🧠 ML-Based Proxy Attendance Detection

This project implements an intelligent proxy attendance detection system using Machine Learning + rule-based logic.
The goal is to identify suspicious attendance patterns while keeping the system explainable, fair, and safe.

🎯 Project Goal

To detect proxy attendance by analysing behavioral patterns instead of blindly accusing students.

✨ The system:
Uses ML only for risk scoring
Applies confidence & rule checks before flagging
Produces human-readable explanations

📁 Dataset Used

The system strictly follows this CSV format 👇

Roll_Number,Date,Time,Subject,Status,Label

🧾 Column Description
Column	Description
🆔 Roll_Number	Unique student ID
📅 Date	Attendance date
⏰ Time	Attendance time (HH:MM)
📘 Subject	Subject name
✅ Status	Present / Absent
🏷️ Label	0 = Normal, 1 = Proxy (ground truth)
⚠️ Column names are standardized internally to avoid errors.

🧠 System Workflow
📂 Attendance CSV
        ⬇️
🛠️ Feature Engineering
        ⬇️
🤖 ML Risk Scoring (Random Forest)
        ⬇️
🧩 Confidence + Rule Validation
        ⬇️
✅ Explainable Proxy Decision

🛠️ Feature Engineering
🔹 Generated Features (used by ML)
Feature	Meaning
⏱️ hour	Hour extracted from attendance time
📊 attendance_frequency	Total attendance count
🔁 same_time_count	Repeated attendance at same time
📚 subject_diversity	Number of unique subjects
📆 day_variance	Number of unique weekdays
⛔ is_fixed_time	Fixed-time attendance flag
👉 These features capture behavior, not identity.

🤖 Machine Learning Model
Model: Random Forest Classifier 🌳
Type: Binary Classification (Normal / Proxy)

Why Random Forest?
Works well on tabular data
Handles non-linear patternS
Easier to interpret than deep models

📌 ML outputs a probability score, not a final accusation.

🧩 Hybrid Decision Logic (Very Important)

Final proxy decision is made only if:
P(proxy) ≥ 0.75
AND
Suspicious behavioral rules are satisfied

🔍 Example Output
{
  "prediction": "Proxy",
  "confidence": 0.87,
  "reasons": [
    "Repeated attendance at same time",
    "Low subject diversity"
  ]
}
✅ Transparent
✅ Explainable
✅ Reviewer-friendly

📂 Project Structure
Smart-Attendance-System/
│
├── data/
│   └── attendance_data.csv 📄
│
├── feature_engineering/
│   └── build_features.py 🛠️
│
├── ml/
│   ├── train_model.py 🤖
│   └── inference.py 🔮
│
├── logic/
│   └── proxy_rules.py 🧩
│
├── run_pipeline.py 🚀
│
├── models/        ⚙️ (auto-generated)
├── artifacts/     ⚙️ (auto-generated)
│
├── requirements.txt 📦
├── .gitignore 🚫
└── README.md 📘

▶️ How to Run the Project

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Complete Pipeline
python run_pipeline.py


✨ This will:
Build features
Train the model (if not already trained)
Perform inference
Apply decision rules
Print final result with confidence & reasons
📈 Model Performance
✅ Accuracy: ~93%
⚖️ Balanced precision & recall
🛡️ Confidence thresholding reduces false positives

⚠️ Important Notes

models/ and artifacts/ are auto-generated
They are ignored using .gitignore
Delete them if features change and retrain
Do not manually edit .pkl files