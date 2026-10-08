"""
Builds a real, executed Jupyter notebook (.ipynb) for the fraud detection
project by running each code cell in a persistent namespace, capturing
stdout and any matplotlib figures produced, and writing valid nbformat v4 JSON.
No internet / nbformat package required - ipynb is just JSON.
"""
import io
import os
import sys
import json
import base64
import contextlib

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CELLS = []  # list of (type, source) where type is 'markdown' or 'code'

def md(text):
    CELLS.append(("markdown", text))

def code(text):
    CELLS.append(("code", text))

# ---------------------------------------------------------------------
md("""# Credit Card Fraud Detection — Machine Learning Project

**Objective:** Build and compare multiple machine learning models to detect fraudulent
credit card transactions in a highly imbalanced dataset, and select the best-performing
model for deployment.

This notebook follows the pipeline: **Data Loading → EDA → Preprocessing → Class
Imbalance Handling (SMOTE) → Model Training → Evaluation & Comparison → Best Model
Selection.**
""")

code("""import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    average_precision_score,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
%matplotlib inline
print("Libraries loaded.")""")

# ---------------------------------------------------------------------
md("## 1. Load the Dataset")

code("""df = pd.read_csv("../data/creditcard_cleaned.csv")
print("Shape:", df.shape)
df.head()""")

code("""print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

class_counts = df["Class"].value_counts()
fraud_pct = 100 * class_counts.get(1, 0) / len(df)
print("\\nClass distribution:\\n", class_counts)
print(f"Fraud percentage: {fraud_pct:.3f}%")""")

# ---------------------------------------------------------------------
md("""## 2. Exploratory Data Analysis (EDA)

The dataset uses PCA-transformed features (`V1`–`V28`) for confidentiality, plus the
raw `Time` and `Amount` columns. We look at class balance, transaction amount patterns,
and which features correlate most with fraud.""")

code("""fig, ax = plt.subplots(figsize=(6, 4.5))
colors = ["#3B82F6", "#EF4444"]
bars = ax.bar(["Legitimate (0)", "Fraud (1)"], class_counts.reindex([0, 1]).values, color=colors)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, str(int(b.get_height())), ha="center", fontweight="bold")
ax.set_title("Class Distribution: Legitimate vs Fraudulent Transactions", fontweight="bold")
ax.set_ylabel("Number of Transactions")
plt.tight_layout()
plt.show()""")

code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df[df.Class == 0]["Amount"], bins=40, ax=axes[0], color="#3B82F6")
axes[0].set_title("Transaction Amount - Legitimate")
sns.histplot(df[df.Class == 1]["Amount"], bins=40, ax=axes[1], color="#EF4444")
axes[1].set_title("Transaction Amount - Fraud")
plt.tight_layout()
plt.show()

print(df.groupby("Class")["Amount"].mean())""")

code("""fig, ax = plt.subplots(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.show()""")

code("""class_corr = corr["Class"].drop("Class").sort_values(key=abs, ascending=False)
top_feats = class_corr.head(10)

fig, ax = plt.subplots(figsize=(7, 5))
colors2 = ["#EF4444" if v < 0 else "#3B82F6" for v in top_feats.values]
ax.barh(top_feats.index[::-1], top_feats.values[::-1], color=colors2[::-1])
ax.set_title("Top 10 Features Correlated with Fraud (Class)", fontweight="bold")
ax.set_xlabel("Correlation coefficient")
plt.tight_layout()
plt.show()

top_feats""")

md("""**Observation:** In the full public Kaggle Credit-Card-Fraud dataset (284,807 rows,
492 frauds), features like `V14`, `V4`, `V12`, `V10`, `V17` show strong correlation
(|r| ≈ 0.3–0.7) with fraud. In this 500-row sample (only **16** fraud examples), the
correlations are much weaker (|r| ≤ 0.16). This is an expected consequence of extreme
class rarity combined with a very small sample — with only 16 positive examples, any
correlation estimate has high variance. This is flagged again in the Limitations
section at the end.""")

# ---------------------------------------------------------------------
md("""## 3. Preprocessing

- Scale `Time` and `Amount` with `StandardScaler` (the `V1`–`V28` columns are already
  PCA-scaled from the original dataset).
- Stratified train/test split (75/25) to preserve the fraud ratio in both sets.""")

code("""X = df.drop(columns=["Class"])
y = df["Class"].values

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
print("Train fraud count:", y_train.sum(), "| Test fraud count:", y_test.sum())

joblib.dump(scaler, "../models/scaler.pkl")""")

# ---------------------------------------------------------------------
md("""## 4. Handling Class Imbalance — SMOTE

Fraud is only ~3.2% of transactions. Training directly on this imbalance biases models
toward always predicting "legitimate". We implement **SMOTE** (Synthetic Minority
Oversampling Technique) manually with `sklearn`'s `NearestNeighbors` (no internet
access in this environment to install `imbalanced-learn`, so SMOTE is implemented
from its published algorithm): for each minority (fraud) sample, generate synthetic
points by interpolating toward its nearest fraud neighbours, **applied to the
training set only** so the test set stays untouched and realistic.""")

code('''def smote_oversample(X_arr, y_arr, k_neighbors=5, random_state=RANDOM_STATE):
    """Minimal SMOTE implementation (Chawla et al., 2002)."""
    rng = np.random.RandomState(random_state)
    X_arr = np.asarray(X_arr)
    minority = X_arr[y_arr == 1]
    majority_count = int((y_arr == 0).sum())
    minority_count = int((y_arr == 1).sum())
    n_to_generate = majority_count - minority_count
    if n_to_generate <= 0 or minority_count < 2:
        return X_arr, y_arr

    k = min(k_neighbors, minority_count - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority)
    _, indices = nn.kneighbors(minority)

    synthetic = []
    for _ in range(n_to_generate):
        i = rng.randint(0, minority_count)
        neighbor_idx = indices[i][rng.randint(1, k + 1)]
        diff = minority[neighbor_idx] - minority[i]
        gap = rng.rand()
        synthetic.append(minority[i] + gap * diff)

    X_res = np.vstack([X_arr, np.array(synthetic)])
    y_res = np.concatenate([y_arr, np.ones(n_to_generate)])
    return X_res, y_res


X_train_bal, y_train_bal = smote_oversample(X_train.values, y_train)
print(f"Before SMOTE -> legit: {(y_train==0).sum()}, fraud: {(y_train==1).sum()}")
print(f"After  SMOTE -> legit: {(y_train_bal==0).sum()}, fraud: {(y_train_bal==1).sum()}")''')

# ---------------------------------------------------------------------
md("""## 5. Model Training

We train six classifiers spanning the families reviewed in the literature survey:
Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting, and a small Neural
Network (MLP). All are trained on the **SMOTE-balanced training set** and evaluated on
the **untouched, realistic test set**.""")

code("""models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=RANDOM_STATE),
}

results = []
roc_data = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan
    ap = average_precision_score(y_test, y_proba)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1").mean()

    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
                     "F1-Score": f1, "ROC-AUC": auc, "Avg Precision (PR-AUC)": ap,
                     "CV F1 (5-fold)": cv_f1})
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)
    print(f"{name:25s} Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)""")

# ---------------------------------------------------------------------
md("## 6. Model Comparison")

code("""results_df""")

code("""metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.15
palette = sns.color_palette("Set2", len(metrics_to_plot))
for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=palette[i])
ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
ax.set_ylim(0, 1.15)
ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=13)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5)
plt.tight_layout()
plt.show()""")

code("""fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - All Models", fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()""")

# ---------------------------------------------------------------------
md("## 7. Best Model Selection")

code("""best_row = results_df.iloc[0]
best_name = best_row["Model"]
best_model = trained_models[best_name]
print("Best model selected (highest F1-Score):", best_name)
print(best_row)""")

code("""y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
ax.set_title(f"Confusion Matrix - {best_name} (Best Model)", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred_best, target_names=["Legit", "Fraud"]))""")

code("""rf_ref = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)
rf_ref.fit(X_scaled, y)
importances = pd.Series(rf_ref.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(importances.index[::-1], importances.values[::-1], color="#6366F1")
ax.set_title("Top Feature Importances - Random Forest (reference model)", fontweight="bold")
plt.tight_layout()
plt.show()""")

code("""import joblib
joblib.dump(best_model, f"../models/best_model_{best_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
print("Model saved to ../models/")""")

# ---------------------------------------------------------------------
md("""## 8. Conclusions & Limitations

**Pipeline delivered:** data loading → EDA → preprocessing → SMOTE class-imbalance
handling → 6 trained classifiers → multi-metric comparison → best-model selection →
saved model artifact — covering every objective in the project proposal.

**Key limitation — dataset size:** this working sample contains only **500
transactions with 16 frauds (3.2%)**. That is far smaller than the full public
Kaggle Credit Card Fraud dataset (284,807 transactions, 492 frauds). With only 16
positive examples split across train/test, both the correlations found in EDA and the
model metrics above have **high statistical variance** — a different random split can
change which model looks "best". This shows up directly in the results: even
Random Forest and Gradient Boosting, normally strong performers on this problem,
scored a 0 recall on the 4-fraud test fold here, and cross-validated F1 across the
whole small sample was low for every model.

**Recommendation:** re-run this exact notebook (`fraud_detection.py` / this notebook,
unchanged) on the full `creditcard.csv` (284,807 rows) for a production-representative
result — the literature survey in the accompanying report shows ROC-AUC in the
0.95–0.99 range and F1 above 0.85 are typical for these same techniques (SMOTE +
Random Forest / Gradient Boosting / ANN) at that scale. The code needs no
modification — only the CSV path changes.
""")

# ---------------------------------------------------------------------
# Execute all cells for real, capture stdout + generated figures
# ---------------------------------------------------------------------

os.makedirs("../outputs/figures", exist_ok=True)
os.makedirs("../models", exist_ok=True)

ns = {}
nb_cells = []
fig_counter = [0]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt

for kind, source in CELLS:
    if kind == "markdown":
        nb_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        })
        continue

    buf = io.StringIO()
    outputs = []
    error = None

    exec_source = source.replace("%matplotlib inline\n", "").replace("%matplotlib inline", "")
    lines = exec_source.rstrip().splitlines()
    last_expr_src = None
    if lines:
        last = lines[-1].strip()
        is_stmt_like = last.startswith(("#", "print(", "for ", "if ", "def ", "class ", "import ", "from ", "with ")) \
            or "=" in last.split("(")[0].replace("==", "") \
            or last.endswith(":")
        if last and not is_stmt_like:
            try:
                compile(last, "<check>", "eval")
                last_expr_src = last
                exec_source = "\n".join(lines[:-1])
            except SyntaxError:
                last_expr_src = None

    try:
        with contextlib.redirect_stdout(buf):
            if exec_source.strip():
                exec(compile(exec_source, "<cell>", "exec"), ns)
            if last_expr_src:
                val = eval(compile(last_expr_src, "<cell-expr>", "eval"), ns)
                if val is not None:
                    outputs.append({
                        "output_type": "execute_result",
                        "execution_count": None,
                        "data": {"text/plain": repr(val).splitlines(keepends=True) or [repr(val)]},
                        "metadata": {},
                    })
            # capture any open figures
            fignums = _plt.get_fignums()
            for num in fignums:
                fig = _plt.figure(num)
                imgbuf = io.BytesIO()
                fig.savefig(imgbuf, format="png", dpi=110, bbox_inches="tight")
                imgbuf.seek(0)
                b64 = base64.b64encode(imgbuf.read()).decode("ascii")
                outputs.append({
                    "output_type": "display_data",
                    "data": {"image/png": b64, "text/plain": ["<Figure>"]},
                    "metadata": {},
                })
                _plt.close(fig)
    except Exception as e:
        error = e

    text = buf.getvalue()
    cell_outputs = []
    if text.strip():
        cell_outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": text.splitlines(keepends=True),
        })
    cell_outputs.extend(outputs)

    if error is not None:
        cell_outputs.append({
            "output_type": "error",
            "ename": type(error).__name__,
            "evalue": str(error),
            "traceback": [f"{type(error).__name__}: {error}"],
        })
        print(f"CELL ERROR:\\n{source}\\n---\\n{error}", file=sys.stderr)

    nb_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": cell_outputs,
        "source": source.splitlines(keepends=True),
    })

notebook = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "../notebooks/Credit_Card_Fraud_Detection.ipynb"
os.makedirs("../notebooks", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"\\nNotebook written to {out_path}")
