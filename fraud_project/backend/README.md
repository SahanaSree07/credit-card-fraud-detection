# Fraud Watch — Full-Stack Fraud Detection App

This adds a **frontend + backend + database** on top of the ML project, so the
trained model is usable through a real web app instead of just a notebook.

- **Frontend:** single-page dashboard (`backend/static/`) — plain HTML/CSS/JS,
  no build step. Dashboard of model metrics, a form to test transactions, and
  a log of past predictions.
- **Backend:** Flask REST API (`backend/app.py`) that loads the trained model
  (`models/best_model_*.pkl`) and scaler, serves predictions, and serves the
  frontend.
- **Database:** SQLite (`backend/fraud_app.db`, created automatically on first
  run) — logs every prediction made through the app: timestamp, amount,
  prediction, probability, true label (if known), and source.

## How to run

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

That's it — no separate frontend build step, no separate database setup. The
SQLite file is created automatically the first time you run the app.

## What you can do in the app

1. **Dashboard** — see the dataset size/fraud rate and every model's Accuracy,
   Precision, Recall, F1, and ROC-AUC, as charts and a table.
2. **Check a transaction:**
   - *Quick check* — type an amount and a time; the other 28 anonymized
     features are filled from a random real transaction (they can't
     realistically be typed by hand — see note below).
   - *Load a real transaction* — pulls a random real row from the dataset
     (with its true label) so you can compare the model's call against
     ground truth.
3. **Prediction log** — every prediction made through the app, pulled live
   from the SQLite database, with a button to clear it.

## API endpoints (if you want to call them directly, or build another frontend)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Confirms the server + model are loaded |
| GET | `/api/stats` | Dataset summary + all models' metrics + usage counts |
| GET | `/api/sample` | Random real transaction + its true label |
| POST | `/api/predict` | Predict on a full 30-feature transaction (`{"features": {...}, "actual_label": 0}`) |
| POST | `/api/predict/quick` | Predict from just `{"amount": ..., "time": ...}` |
| GET | `/api/history?limit=50` | Recent logged predictions |
| DELETE | `/api/history` | Clear the log |

## Why "Quick check" doesn't ask for V1–V28

The 28 PCA-anonymized features (`V1`...`V28`) are the output of a
confidentiality transform on the original bank data — they have no real-world
meaning a person could type in (unlike Amount or Time), and the source Kaggle
dataset doesn't publish the transform needed to compute them from raw card
details. So the demo form takes the two fields a person actually has
(Amount, Time) and pairs them with a real reference transaction's V1–V28,
which is realistic for demoing the model without pretending a user could
hand-enter anonymized features. Loading a full real transaction — features
and all — is what "Load a real transaction" does, and is the more rigorous way
to check the model's accuracy against ground truth.

## Project structure (full picture)

```
fraud_project/
├── data/            # dataset
├── models/          # trained model + scaler (used by the backend)
├── notebooks/       # ML notebook (Part 1 deliverable)
├── outputs/         # figures, report, metrics (used by the dashboard)
├── src/             # ML pipeline scripts
└── backend/         # <-- NEW: the web app
    ├── app.py               # Flask API + server
    ├── requirements.txt
    ├── fraud_app.db          # created automatically on first run
    └── static/
        ├── index.html
        ├── style.css
        └── app.js
```

## Notes for a viva / demo

- This is a Flask **development** server — fine for a project demo, but a
  real deployment would use gunicorn/uwsgi behind nginx, which is a one-line
  change (`gunicorn app:app`) once you're ready.
- The database only logs predictions made *through the app* — it doesn't
  duplicate the training dataset. That's intentional: it's an audit trail of
  the model in use, not a copy of the training data.
- Remember the same limitation noted in the ML report: the training data is
  only 500 rows with 16 fraud cases, so live predictions here are for
  **demonstrating the working system**, not for judging real-world accuracy.
