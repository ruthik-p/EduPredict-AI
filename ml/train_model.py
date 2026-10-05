"""
train_model.py
--------------
Trains, evaluates, and saves:
  - DecisionTreeClassifier  →  performance_category
  - DecisionTreeRegressor   →  final_score

ML Pipeline:
  1. Load synthetic dataset
  2. Data preprocessing (missing-value check, duplicate check, range validation,
     categorical encoding, train/test split — no leakage)
  3. Train models on training set only
  4. Evaluate on held-out test set
  5. Save models as .pkl files
  6. Save metadata to ml/model_info.json
  7. Save evaluation results to documentation/MODEL_EVALUATION.md
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(BASE_DIR, "data", "student_data.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "ml", "models")
MODEL_INFO    = os.path.join(BASE_DIR, "ml", "model_info.json")
EVAL_MD       = os.path.join(BASE_DIR, "documentation", "MODEL_EVALUATION.md")

FEATURE_COLS = [
    "attendance_pct",
    "study_hours_per_week",
    "assignment_avg",
    "internal_marks",
    "prev_exam_score",
    "assignment_completion",
    "participation_level",
    "num_backlogs",
]

CATEGORY_ORDER = ["Needs Improvement", "Average", "Good", "Excellent"]

# ── 1. Load dataset ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Dataset not found at {path}. Run data/generate_dataset.py first.")
    df = pd.read_csv(path)
    print(f"  Loaded dataset        : {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ── 2. Preprocess ───────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("\n[2] Preprocessing")

    # Missing values
    missing = df.isnull().sum().sum()
    print(f"  Missing values        : {missing}")
    if missing > 0:
        df = df.dropna()
        print(f"  Rows after drop NA    : {len(df)}")

    # Duplicates
    dupes = df.duplicated().sum()
    print(f"  Duplicate rows        : {dupes}")
    if dupes > 0:
        df = df.drop_duplicates()
        print(f"  Rows after dedup      : {len(df)}")

    # Range validation
    range_rules = {
        "attendance_pct":        (0, 100),
        "study_hours_per_week":  (0, 40),
        "assignment_avg":        (0, 100),
        "internal_marks":        (0, 50),
        "prev_exam_score":       (0, 100),
        "assignment_completion": (0, 100),
        "participation_level":   (0, 2),
        "num_backlogs":          (0, 10),
        "final_score":           (0, 100),
    }
    violations = 0
    for col, (lo, hi) in range_rules.items():
        if col in df.columns:
            bad = ((df[col] < lo) | (df[col] > hi)).sum()
            if bad > 0:
                print(f"  [WARN] {col}: {bad} out-of-range values — clipping")
                df[col] = df[col].clip(lo, hi)
                violations += bad
    if violations == 0:
        print("  Range validation      : all values within expected bounds")

    # Encode categorical target  (participation_level is already integer 0/1/2)
    le = LabelEncoder()
    le.fit(CATEGORY_ORDER)
    df["category_encoded"] = le.transform(df["performance_category"])

    return df, le


# ── 3. Split ─────────────────────────────────────────────────────────────────
def split(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y_clf = df["category_encoded"]
    y_reg = df["final_score"]

    # Stratify on classification target to preserve class proportions
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg,
        test_size=0.20,
        random_state=42,
        stratify=y_clf,
    )
    print(f"\n[3] Train/Test Split")
    print(f"  Training samples      : {len(X_train)}")
    print(f"  Test samples          : {len(X_test)}")
    return X_train, X_test, yc_train, yc_test, yr_train, yr_test


# ── 4. Train & Evaluate Classification ──────────────────────────────────────────
def train_classifier(X_train, X_test, yc_train, yc_test, le):
    print("\n[4] Classification — DecisionTreeClassifier")

    clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    clf.fit(X_train, yc_train)
    yc_pred = clf.predict(X_test)

    acc  = accuracy_score(yc_test, yc_pred)
    prec = precision_score(yc_test, yc_pred, average="weighted", zero_division=0)
    rec  = recall_score(yc_test, yc_pred, average="weighted", zero_division=0)
    f1   = f1_score(yc_test, yc_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(yc_test, yc_pred)
    cr   = classification_report(yc_test, yc_pred,
                                  target_names=le.classes_,
                                  zero_division=0)

    print(f"  Accuracy              : {acc:.4f}")
    print(f"  Precision (weighted)  : {prec:.4f}")
    print(f"  Recall (weighted)     : {rec:.4f}")
    print(f"  F1-score (weighted)   : {f1:.4f}")
    print(f"\n  Classification Report:\n{cr}")
    print(f"  Confusion Matrix:\n{cm}")

    fi = dict(zip(FEATURE_COLS, clf.feature_importances_.round(4)))
    print(f"\n  Feature Importances (classifier):")
    for k, v in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {k:<28}: {v}")

    metrics = {"accuracy": round(acc,4), "precision_weighted": round(prec,4),
               "recall_weighted": round(rec,4), "f1_weighted": round(f1,4),
               "confusion_matrix": cm.tolist(),
               "classification_report": cr}
    return clf, metrics, fi


# ── 5. Train & Evaluate Regression ──────────────────────────────────────────────
def train_regressor(X_train, X_test, yr_train, yr_test):
    print("\n[5] Regression — DecisionTreeRegressor")

    reg = DecisionTreeRegressor(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    reg.fit(X_train, yr_train)
    yr_pred = reg.predict(X_test)

    mae  = mean_absolute_error(yr_test, yr_pred)
    mse  = mean_squared_error(yr_test, yr_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(yr_test, yr_pred)

    print(f"  MAE                   : {mae:.4f}")
    print(f"  MSE                   : {mse:.4f}")
    print(f"  RMSE                  : {rmse:.4f}")
    print(f"  R²                    : {r2:.4f}")

    fi = dict(zip(FEATURE_COLS, reg.feature_importances_.round(4)))
    print(f"\n  Feature Importances (regressor):")
    for k, v in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {k:<28}: {v}")

    metrics = {"mae": round(mae,4), "mse": round(mse,4),
               "rmse": round(rmse,4), "r2": round(r2,4)}
    return reg, metrics, fi


# ── 6. Save models ───────────────────────────────────────────────────────────────
def save_models(clf, reg, le):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "classifier.pkl"))
    joblib.dump(reg, os.path.join(MODELS_DIR, "regressor.pkl"))
    joblib.dump(le,  os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"\n[6] Models saved to {MODELS_DIR}")
    print(f"  classifier.pkl, regressor.pkl, label_encoder.pkl")


# ── 7. Save model_info.json ──────────────────────────────────────────────────────
def save_model_info(clf_metrics, reg_metrics, clf_fi, reg_fi):
    info = {
        "project":          "EduPredict AI",
        "trained_at":       datetime.now().isoformat(timespec="seconds"),
        "dataset":          "data/student_data.csv (synthetic, 500 records)",
        "random_seed":      42,
        "train_test_split": "80% train / 20% test (stratified on category)",
        "features":         FEATURE_COLS,
        "classifier": {
            "algorithm":   "DecisionTreeClassifier",
            "max_depth":   6,
            "min_samples_split": 10,
            "min_samples_leaf":  5,
            "metrics":     clf_metrics,
            "feature_importances": clf_fi,
        },
        "regressor": {
            "algorithm":  "DecisionTreeRegressor",
            "max_depth":  6,
            "min_samples_split": 10,
            "min_samples_leaf":  5,
            "metrics":    reg_metrics,
            "feature_importances": reg_fi,
        },
        "categories": CATEGORY_ORDER,
        "note": (
            "All models trained on SYNTHETIC data only. "
            "Not intended for real academic evaluation."
        ),
    }
    # confusion_matrix is a list-of-lists — already JSON-serialisable
    with open(MODEL_INFO, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  model_info.json saved to {MODEL_INFO}")


# ── 8. Save MODEL_EVALUATION.md ─────────────────────────────────────────────────
def save_eval_md(clf_metrics, reg_metrics, clf_fi, reg_fi, le):
    cm = clf_metrics["confusion_matrix"]
    cats = list(le.classes_)
    cm_header = "| |" + "|".join(f" Pred: {c} " for c in cats) + "|"
    cm_sep    = "|---|" + "|".join(["---|"] * len(cats))
    cm_rows   = "\n".join(
        "| **Act: " + cats[i] + "** |" + "|".join(str(v) for v in row) + "|"
        for i, row in enumerate(cm)
    )

    fi_clf_rows = "\n".join(
        f"| {k} | {v} |"
        for k, v in sorted(clf_fi.items(), key=lambda x: -x[1])
    )
    fi_reg_rows = "\n".join(
        f"| {k} | {v} |"
        for k, v in sorted(reg_fi.items(), key=lambda x: -x[1])
    )

    md = f"""# EduPredict AI — Model Evaluation Report

> Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

> **Dataset Notice:** All results are based on a **synthetic dataset** of 500
> programmatically generated records. This dataset contains **no real student
> names, IDs, or personal information**. Results are for educational demonstration
> only and should not be used for real academic decision-making.

---

## Dataset Summary

| Property | Value |
|---|---|
| Total records | 500 |
| Train samples | 400 (80%) |
| Test samples | 100 (20%) |
| Features | {len(FEATURE_COLS)} |
| Split strategy | Stratified by performance category |
| Random seed | 42 |

---

## Classification Model — DecisionTreeClassifier

### Hyperparameters

| Parameter | Value |
|---|---|
| max_depth | 6 |
| min_samples_split | 10 |
| min_samples_leaf | 5 |
| random_state | 42 |

### Overall Metrics (Weighted Average)

| Metric | Score |
|---|---|
| Accuracy | {clf_metrics['accuracy']} |
| Precision | {clf_metrics['precision_weighted']} |
| Recall | {clf_metrics['recall_weighted']} |
| F1-score | {clf_metrics['f1_weighted']} |

### Confusion Matrix

{cm_header}
{cm_sep}
{cm_rows}

### Feature Importances (Classifier)

| Feature | Importance |
|---|---|
{fi_clf_rows}

---

## Regression Model — DecisionTreeRegressor

### Hyperparameters

| Parameter | Value |
|---|---|
| max_depth | 6 |
| min_samples_split | 10 |
| min_samples_leaf | 5 |
| random_state | 42 |

### Metrics

| Metric | Value |
|---|---|
| MAE | {reg_metrics['mae']} |
| MSE | {reg_metrics['mse']} |
| RMSE | {reg_metrics['rmse']} |
| R² | {reg_metrics['r2']} |

### Feature Importances (Regressor)

| Feature | Importance |
|---|---|
{fi_reg_rows}

---

## Notes

- Decision Trees do not require feature scaling; **no StandardScaler was applied**.
- The classification target was label-encoded using `LabelEncoder` with class order:
  `{CATEGORY_ORDER}`.
- `participation_level` is already an ordinal integer (0=Low, 1=Medium, 2=High).
- All preprocessing (missing-value handling, duplicate removal, range clipping)
  was applied **before** the train/test split to prevent data leakage.
- Models were not artificially tuned for maximum accuracy — results reflect
  genuine test-set performance on synthetic data.
"""
    with open(EVAL_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  MODEL_EVALUATION.md saved to {EVAL_MD}")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  EduPredict AI — Model Training Pipeline")
    print("=" * 55)

    print("\n[1] Loading dataset")
    df = load_data(DATA_PATH)

    df, le = preprocess(df)

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = split(df)

    clf, clf_metrics, clf_fi = train_classifier(X_train, X_test, yc_train, yc_test, le)

    reg, reg_metrics, reg_fi = train_regressor(X_train, X_test, yr_train, yr_test)

    save_models(clf, reg, le)
    save_model_info(clf_metrics, reg_metrics, clf_fi, reg_fi)
    save_eval_md(clf_metrics, reg_metrics, clf_fi, reg_fi, le)

    print("\n" + "=" * 55)
    print("  Training complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
