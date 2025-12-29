## Machine Learning Model – Proxy Attendance Detection

This system uses a supervised Machine Learning approach to detect proxy attendance patterns.

### Model Used
- Random Forest Classifier

### Why Random Forest?
- Handles structured tabular data effectively
- Learns complex attendance behavior patterns
- Reduces overfitting compared to a single decision tree
- Provides probability scores for confidence estimation

### Feature Engineering
Instead of using raw attendance records, behavioral features are extracted:
- Attendance frequency per student
- Repeated attendance timings
- Subject diversity
- Day-wise attendance variance
- Attendance status (Present / Absent)

These features help the model identify suspicious proxy-like behavior.

### Confidence Score
The model outputs both:
- Prediction (Normal / Proxy)
- Confidence score (0–1)

This allows the system to flag:
- High-risk proxy cases
- Medium-risk suspicious cases
- Low-risk normal attendance

### Model Performance
- Accuracy: > 90%
- Cross-validation accuracy: Stable across folds

The high accuracy is expected due to clearly defined patterns in the synthetic dataset.
