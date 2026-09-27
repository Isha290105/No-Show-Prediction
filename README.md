---
title: Hospital No-Show Predictor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 🏥 Hospital Appointment No-Show Predictor & Overbooking Simulator

A Streamlit frontend for a hospital no-show prediction & overbooking
simulation project.

## Pages

- **Overview** — project summary and headline findings
- **Insights (EDA)** — interactive charts: weekday effect, SMS reminders,
  lead time, weather, chronic conditions, correlations
- **Predict a No-Show** — fill in one appointment's details and get a
  calibrated no-show risk score
- **Batch Prediction** — upload a CSV of appointments and download scored
  results
- **Overbooking Simulator** — adjust capacity / overbooking % and compare
  "No Overbooking" vs "Fixed %" vs "Smart (model-based)" policies
- **Model Performance** — ROC curve, confusion matrix, feature importance

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```


## Folder structure

```
.
├── app.py
├── requirements.txt
├── README.md               (this file — doubles as the Space card)
└── data/
    ├── patients.csv
    ├── appointments.csv
    ├── best_model.pkl              (class-balanced RF, for classification)
    ├── calibrated_model.pkl        (calibrated RF, used for probabilities)
    ├── test_predictions.csv        (held-out test set with predictions)
    ├── policy_comparison_summary.csv
    └── smart_policy_daily_results.csv
```