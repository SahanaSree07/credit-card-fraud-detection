# Credit Card Fraud Detection — Full ML Project

A complete, working machine learning pipeline for detecting fraudulent credit
card transactions, built around the objectives in the project proposal
(EDA → preprocessing → class-imbalance handling → multi-model training →
comparison → best-model selection).

## Folder structure

```
fraud_project/
├── data/
│   └── creditcard_cleaned.csv        # your dataset (500 rows, 16 fraud)
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb   # main deliverable — fully executed notebook
├── src/
│   ├── fraud_detection.py            # same pipeline as a standalone script
│   ├── build_notebook.py             # regenerates the notebook (dev tool)
│   └── build_report.js               # regenerates the Word report (dev tool)
├── outputs/
│   ├── Fraud_Detection_Project_Report.docx   # full written report
│   ├── figures/                      # all charts (PNG)
│   ├── model_comparison.csv          # metrics table for all 6 models
│   ├── summary.json                  # best model + its metrics
│   └── best_model_classification_report.txt
├── models/
│   ├── scaler.pkl                    # fitted StandardScaler
│   └── best_model_*.pkl              # the selected best model, ready to load
├── backend/                          # frontend + backend + database (web app)
│   ├── app.py, requirements.txt, README.md
│   └── static/ (index.html, style.css, app.js)
└── requirements.txt
```

## How to run

```bash
cd src
pip install -r ../requirements.txt
python fraud_detection.py
```

This regenerates everything in `outputs/` and `models/` from the CSV in `data/`.
Or open `notebooks/Credit_Card_Fraud_Detection.ipynb` directly — it already
contains the full run with all outputs and charts.

## Using the saved model on new transactions

```python
import joblib
import pandas as pd

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/best_model_K-Nearest_Neighbors.pkl")

new_txn = pd.DataFrame([...])  # same columns as training data, minus 'Class'
new_txn[["Time", "Amount"]] = scaler.transform(new_txn[["Time", "Amount"]])
prediction = model.predict(new_txn)          # 0 = legit, 1 = fraud
probability = model.predict_proba(new_txn)   # confidence scores
```

## Full-stack web app (frontend + backend + database)

There's now a complete web app on top of the ML pipeline — see
`backend/README.md` for details. Quick start:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. It has a dashboard, a page to test
transactions against the trained model, and a SQLite-backed prediction log.

## Important note on this dataset

The provided `creditcard_cleaned.csv` has only **500 transactions and 16
fraud cases**. That's a very small sample for supervised fraud detection —
metrics reported here have high variance, and this is discussed honestly in
Section 5 ("Limitations") of the report. The exact same code, unmodified,
will produce much stronger and more stable results if you point `DATA_PATH`
in `fraud_detection.py` at the full public Kaggle Credit Card Fraud dataset
(284,807 rows, 492 frauds) — just swap the CSV file.
