"""
Credit Card Fraud Detection - Full ML Pipeline
================================================
Loads a PCA-transformed credit card transactions dataset, performs EDA,
handles severe class imbalance, trains and compares multiple ML models,
and selects the best model for deployment.

Run:  python fraud_detection.py
Outputs are written to ../outputs/ (figures, metrics, logs) and
../models/ (the saved best model).
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
    average_precision_score, precision_recall_curve,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "../data/creditcard_cleaned.csv"
FIG_DIR = "../outputs/figures"
RESULTS_DIR = "../outputs"
MODEL_DIR = "../models"

import os
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def log(msg):
    print(msg)


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
log("=" * 70)
log("STEP 1: LOAD DATA")
log("=" * 70)

df = pd.read_csv(DATA_PATH)
log(f"Shape: {df.shape}")
log(f"Columns: {list(df.columns)}")
log(f"Missing values total: {df.isnull().sum().sum()}")
log(f"Duplicate rows: {df.duplicated().sum()}")

df = df.drop_duplicates().reset_index(drop=True)

class_counts = df["Class"].value_counts()
fraud_pct = 100 * class_counts.get(1, 0) / len(df)
log(f"\nClass distribution:\n{class_counts}")
log(f"Fraud percentage: {fraud_pct:.3f}%")

with open(f"{RESULTS_DIR}/data_summary.json", "w") as f:
    json.dump({
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_legit": int(class_counts.get(0, 0)),
        "n_fraud": int(class_counts.get(1, 0)),
        "fraud_pct": round(fraud_pct, 4),
    }, f, indent=2)

# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 2: EXPLORATORY DATA ANALYSIS")
log("=" * 70)

# 2a. Class balance plot
fig, ax = plt.subplots(figsize=(6, 4.5))
colors = ["#3B82F6", "#EF4444"]
bars = ax.bar(["Legitimate (0)", "Fraud (1)"], class_counts.reindex([0, 1]).values, color=colors)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, str(int(b.get_height())),
            ha="center", fontweight="bold")
ax.set_title("Class Distribution: Legitimate vs Fraudulent Transactions", fontweight="bold")
ax.set_ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_distribution.png", dpi=150)
plt.close()

# 2b. Amount distribution by class
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df[df.Class == 0]["Amount"], bins=40, ax=axes[0], color="#3B82F6")
axes[0].set_title("Transaction Amount - Legitimate")
sns.histplot(df[df.Class == 1]["Amount"], bins=40, ax=axes[1], color="#EF4444")
axes[1].set_title("Transaction Amount - Fraud")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_amount_distribution.png", dpi=150)
plt.close()

# 2c. Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()

# 2d. Features most correlated with Class
class_corr = corr["Class"].drop("Class").sort_values(key=abs, ascending=False)
top_feats = class_corr.head(10)
fig, ax = plt.subplots(figsize=(7, 5))
colors2 = ["#EF4444" if v < 0 else "#3B82F6" for v in top_feats.values]
ax.barh(top_feats.index[::-1], top_feats.values[::-1], color=colors2[::-1])
ax.set_title("Top 10 Features Correlated with Fraud (Class)", fontweight="bold")
ax.set_xlabel("Correlation coefficient")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_top_correlated_features.png", dpi=150)
plt.close()

log(f"Top features correlated with fraud:\n{top_feats}")

# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 3: PREPROCESSING")
log("=" * 70)

X = df.drop(columns=["Class"])
y = df["Class"].values

# Scale Time and Amount (V1-V28 are already PCA-scaled)
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
log(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
log(f"Train fraud count: {y_train.sum()} | Test fraud count: {y_test.sum()}")

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")


# ---------------------------------------------------------------------------
# 4. HANDLE CLASS IMBALANCE (custom SMOTE — no internet access for imblearn)
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 4: HANDLING CLASS IMBALANCE (SMOTE oversampling)")
log("=" * 70)


def smote_oversample(X_arr, y_arr, k_neighbors=5, random_state=RANDOM_STATE):
    """Minimal SMOTE implementation: synthesize new minority samples by
    interpolating between each minority sample and one of its k nearest
    minority neighbours. Only the TRAINING set should ever be passed here."""
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
log(f"Before SMOTE -> legit: {(y_train == 0).sum()}, fraud: {(y_train == 1).sum()}")
log(f"After  SMOTE -> legit: {(y_train_bal == 0).sum()}, fraud: {(y_train_bal == 1).sum()}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(["Before SMOTE\n(fraud)", "After SMOTE\n(fraud)"],
       [int((y_train == 1).sum()), int((y_train_bal == 1).sum())],
       color=["#F59E0B", "#10B981"])
ax.set_title("Effect of SMOTE Oversampling on Training Data", fontweight="bold")
ax.set_ylabel("Number of Fraud Samples")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_smote_effect.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. MODEL TRAINING
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5: MODEL TRAINING")
log("=" * 70)

models = {
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
    log(f"\nTraining: {name}")
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

    # 5-fold stratified CV F1 on the (unbalanced) full scaled dataset for robustness
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1").mean()

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1, "ROC-AUC": auc, "Avg Precision (PR-AUC)": ap,
        "CV F1 (5-fold)": cv_f1,
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    log(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  "
        f"F1={f1:.4f}  ROC-AUC={auc:.4f}")
    log(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
log("\n" + "=" * 70)
log("MODEL COMPARISON TABLE (sorted by F1-Score)")
log("=" * 70)
log(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. VISUALIZE MODEL COMPARISON
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 6: VISUALIZING MODEL COMPARISON")
log("=" * 70)

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
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
plt.savefig(f"{FIG_DIR}/06_model_comparison.png", dpi=150)
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - All Models", fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. SELECT BEST MODEL
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 7: SELECT BEST MODEL")
log("=" * 70)

best_row = results_df.iloc[0]
best_name = best_row["Model"]
best_model = trained_models[best_name]
log(f"Best model selected: {best_name}")
log(f"Metrics: {best_row.to_dict()}")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
ax.set_title(f"Confusion Matrix - {best_name} (Best Model)", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_best_model_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance if available
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(importances.index[::-1], importances.values[::-1], color="#6366F1")
    ax.set_title(f"Top Feature Importances - {best_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_feature_importance.png", dpi=150)
    plt.close()

joblib.dump(best_model, f"{MODEL_DIR}/best_model_{best_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")

report_text = classification_report(y_test, y_pred_best, target_names=["Legit", "Fraud"])
with open(f"{RESULTS_DIR}/best_model_classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n\n{report_text}")
log(f"\nClassification Report:\n{report_text}")

with open(f"{RESULTS_DIR}/summary.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in best_row.to_dict().items()},
    }, f, indent=2)

log("\n" + "=" * 70)
log("PIPELINE COMPLETE. Outputs saved to ../outputs and ../models")
log("=" * 70)
