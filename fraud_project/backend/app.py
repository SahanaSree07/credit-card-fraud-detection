"""
Flask backend for the Credit Card Fraud Detection project.

Serves:
  - A REST API for making predictions (single or sample-from-dataset), and
    reading prediction history / dashboard stats.
  - The static frontend (single-page dashboard).
  - A SQLite database that logs every prediction made through the API.

Run:
    cd backend
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
import os
import json
import sqlite3
import random
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "creditcard_cleaned.csv")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
DB_PATH = os.path.join(BASE_DIR, "fraud_app.db")

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------------------------------------------------------------------------
# Load model artifacts + reference dataset once at startup
# ---------------------------------------------------------------------------
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("best_model_")]
if not model_files:
    raise FileNotFoundError("No best_model_*.pkl found in models/. Run src/fraud_detection.py first.")
MODEL_PATH = os.path.join(MODELS_DIR, model_files[0])
MODEL_NAME = model_files[0].replace("best_model_", "").replace(".pkl", "").replace("_", " ")

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

df_reference = pd.read_csv(DATA_PATH)
FEATURE_COLUMNS = [c for c in df_reference.columns if c != "Class"]

print(f"Loaded model: {MODEL_NAME}")
print(f"Loaded {len(df_reference)} reference rows with {len(FEATURE_COLUMNS)} features")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            time_val REAL,
            amount REAL,
            features_json TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            actual_label INTEGER,
            model_name TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual'
        )
    """)
    conn.commit()
    conn.close()


init_db()


def run_prediction(feature_dict):
    """feature_dict must contain every column in FEATURE_COLUMNS."""
    row = pd.DataFrame([[feature_dict[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])
    row_values = row.values  # avoids sklearn's "fitted without feature names" warning
    pred = int(model.predict(row_values)[0])
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(row_values)[0][1])
    else:
        proba = float(pred)
    return pred, proba


def log_prediction(feature_dict, pred, proba, actual_label=None, source="manual"):
    db = get_db()
    db.execute(
        """INSERT INTO predictions
           (created_at, time_val, amount, features_json, prediction, probability, actual_label, model_name, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            feature_dict.get("Time"),
            feature_dict.get("Amount"),
            json.dumps(feature_dict),
            pred,
            proba,
            actual_label,
            MODEL_NAME,
            source,
        ),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Routes - frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------------------
# Routes - API
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/api/sample", methods=["GET"])
def api_sample():
    """Return a random real transaction from the dataset (features only + its
    true label), so the demo UI doesn't require hand-typing 28 PCA values."""
    row = df_reference.sample(1).iloc[0]
    feature_dict = {c: float(row[c]) for c in FEATURE_COLUMNS}
    return jsonify({"features": feature_dict, "actual_label": int(row["Class"])})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    features = payload.get("features")
    actual_label = payload.get("actual_label")
    source = payload.get("source", "manual")

    if not features:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        return jsonify({"error": f"Missing feature values: {missing}"}), 400

    try:
        pred, proba = run_prediction(features)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    log_prediction(features, pred, proba, actual_label, source)

    return jsonify({
        "prediction": pred,
        "prediction_label": "Fraud" if pred == 1 else "Legitimate",
        "probability": round(proba, 4),
        "model": MODEL_NAME,
    })


@app.route("/api/predict/quick", methods=["POST"])
def api_predict_quick():
    """Simplified predict for the demo form: user supplies only Time & Amount,
    the remaining PCA features are drawn from a random real reference row so
    the request is still a realistic, valid input vector."""
    payload = request.get_json(force=True)
    time_val = float(payload.get("time", 0))
    amount = float(payload.get("amount", 0))

    base_row = df_reference.sample(1).iloc[0]
    features = {c: float(base_row[c]) for c in FEATURE_COLUMNS}
    features["Time"] = time_val
    features["Amount"] = amount

    pred, proba = run_prediction(features)
    log_prediction(features, pred, proba, actual_label=None, source="quick_form")

    return jsonify({
        "prediction": pred,
        "prediction_label": "Fraud" if pred == 1 else "Legitimate",
        "probability": round(proba, 4),
        "model": MODEL_NAME,
        "note": "V1-V28 were sampled from a random real reference transaction; only Time and Amount reflect your input.",
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    limit = int(request.args.get("limit", 50))
    db = get_db()
    rows = db.execute(
        "SELECT id, created_at, time_val, amount, prediction, probability, actual_label, model_name, source "
        "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    db = get_db()
    db.execute("DELETE FROM predictions")
    db.commit()
    return jsonify({"status": "cleared"})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Dataset summary + model comparison metrics, for the dashboard."""
    with open(os.path.join(OUTPUTS_DIR, "data_summary.json")) as f:
        data_summary = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, "model_comparison.json")) as f:
        model_comparison = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, "summary.json")) as f:
        best_summary = json.load(f)

    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    frauds_flagged = db.execute("SELECT COUNT(*) c FROM predictions WHERE prediction=1").fetchone()["c"]

    return jsonify({
        "dataset": data_summary,
        "model_comparison": model_comparison,
        "best_model": best_summary,
        "app_usage": {"total_predictions_logged": total, "frauds_flagged": frauds_flagged},
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
