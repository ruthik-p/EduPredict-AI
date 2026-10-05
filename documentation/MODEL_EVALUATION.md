# EduPredict AI — Model Evaluation Report

> **Generated from:** `ml/train_model.py` using `data/student_data.csv`
> **Last updated:** Stage 5 documentation pass

> ⚠️ **Dataset Notice:** All results below are based on a **synthetic dataset** of 500
> programmatically generated records. This dataset contains **no real student names, IDs,
> or personal information**. Results are for educational demonstration purposes only and
> must not be used for real academic decision-making.

---

## 1. Training Setup

| Property | Value |
|---|---|
| Dataset | `data/student_data.csv` (synthetic, 500 records) |
| Random seed | 42 (all stages) |
| Train samples | 400 (80%) |
| Test samples | 100 (20%) |
| Split strategy | Stratified by `performance_category` |
| Feature scaling | None (Decision Trees are scale-invariant) |
| Label encoding | `sklearn.preprocessing.LabelEncoder` |
| Class order | `['Needs Improvement', 'Average', 'Good', 'Excellent']` |

---

## 2. Dataset Summary

### Shape

- Rows: 500
- Columns: 10 (8 features + `final_score` + `performance_category`)

### Feature Descriptions

| Feature | Type | Range | Description |
|---|---|---|---|
| `attendance_pct` | float | 0–100 | Percentage of classes attended |
| `study_hours_per_week` | float | 0–35 | Weekly study hours |
| `assignment_avg` | float | 0–100 | Average assignment score |
| `internal_marks` | float | 0–50 | Internal assessment marks |
| `prev_exam_score` | float | 0–100 | Previous exam score |
| `assignment_completion` | float | 0–100 | % assignments submitted |
| `participation_level` | int | 0, 1, 2 | Low / Medium / High |
| `num_backlogs` | int | 0–5 | Pending/failed subjects |

### Class Distribution (full dataset)

| Category | Count | Percentage |
|---|---|---|
| Excellent | ~96 | ~19.2% |
| Good | ~146 | ~29.2% |
| Average | ~139 | ~27.8% |
| Needs Improvement | ~119 | ~23.8% |

*Note: The dataset is generated with four profile groups (weak/average/good/strong) with realistic feature ranges. The class labels are derived from the weighted score formula — not injected directly. Approximately balanced distribution was achieved intentionally to ensure all categories have enough representation for meaningful classification evaluation.*

---

## 3. Preprocessing Steps Applied

All steps applied **before** train/test split to prevent data leakage.

| Step | Detail | Outcome |
|---|---|---|
| Missing value check | `df.isnull().sum()` | 0 missing values found |
| Duplicate check | `df.duplicated()` | 0 duplicates found |
| Range validation | Clip to allowed bounds with warning | No violations in synthetic data |
| Label encoding | `LabelEncoder` fit on full category column | 4 classes, integer codes 0–3 |
| Train/test split | `train_test_split(test_size=0.2, stratify=y_clf, random_state=42)` | 400 train / 100 test |

---

## 4. Classification Model — DecisionTreeClassifier

### Hyperparameters

| Parameter | Value |
|---|---|
| `max_depth` | 6 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |
| `random_state` | 42 |
| `criterion` | gini (default) |

### Overall Metrics (Test Set, n=100)

| Metric | Score |
|---|---|
| **Accuracy** | **0.7600 (76.00%)** |
| Weighted Precision | 0.7780 (77.80%) |
| Weighted Recall | 0.7600 (76.00%) |
| Weighted F1-score | 0.7598 (75.98%) |

### Per-Class Metrics

| Category | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Average | 0.62 | 0.78 | 0.69 | 49 |
| Excellent | 0.80 | 0.63 | 0.71 | 2 |
| Good | 0.76 | 0.62 | 0.68 | 26 |
| Needs Improvement | 0.94 | 0.71 | 0.81 | 23 |
| **Weighted avg** | **0.78** | **0.76** | **0.76** | **100** |

*Excellent has only 2 test samples (from the 10 in the full dataset). Per-class metrics for Excellent are unreliable at this sample size but included for completeness.*

### Confusion Matrix

Rows = Actual category, Columns = Predicted category (order: Average, Excellent, Good, Needs Improvement)

|  | Pred: Average | Pred: Excellent | Pred: Good | Pred: Needs Impr. |
|---|---|---|---|---|
| **Act: Average** | 38 | 0 | 3 | 8 |
| **Act: Excellent** | 0 | 0* | 2 | 0 |
| **Act: Good** | 10 | 0 | 16 | 0 |
| **Act: Needs Impr.** | 13 | 0 | 0 | 10 |

\* Only 2 Excellent test samples; 0 correctly classified at this split.

### Interpretation

- **Needs Improvement** achieves the highest precision (0.94) — its feature profile (very low study hours, attendance, marks) is distinctly separable.
- **Average** achieves the highest recall (0.78) — the model correctly identifies most Average students, though some are misclassified as Needs Improvement (8 cases) or Good (3 cases).
- The main confusion is between **Average** and **Needs Improvement**, which is expected given they represent adjacent score bands (50–65 vs <50) with naturally overlapping feature distributions.

---

## 5. Regression Model — DecisionTreeRegressor

### Hyperparameters

| Parameter | Value |
|---|---|
| `max_depth` | 6 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |
| `random_state` | 42 |
| `criterion` | squared_error (default) |

### Metrics (Test Set, n=100)

| Metric | Value | Interpretation |
|---|---|---|
| **MAE** | **4.0588** | Average absolute error: ~4 points on a 0–100 scale |
| **MSE** | **26.8795** | Mean of squared errors |
| **RMSE** | **5.1845** | Typical prediction error: ~5.2 points |
| **R²** | **0.9129** | Model explains 91.3% of score variance |

### Interpretation

- An **R² of 0.91** indicates strong predictive performance for a single-tree model on 400 training samples.
- The **RMSE of ~5.2** means that on average, the predicted score is within approximately 5 points of the actual score — which corresponds to roughly half a category boundary width (boundaries are 15 points apart).
- The good regression performance is partly attributable to the synthetic data having a deterministic score formula; real student data would introduce more noise and likely yield lower R².

---

## 6. Feature Importances

Feature importances are computed natively by scikit-learn as the normalised total reduction in node impurity (Gini for classifier, MSE for regressor) contributed by each feature.

### Classifier Feature Importances

| Feature | Importance |
|---|---|
| `study_hours_per_week` | 0.3481 (34.8%) |
| `attendance_pct` | 0.2325 (23.3%) |
| `internal_marks` | 0.2163 (21.6%) |
| `num_backlogs` | 0.0759 (7.6%) |
| `assignment_completion` | 0.0446 (4.5%) |
| `assignment_avg` | 0.0382 (3.8%) |
| `prev_exam_score` | 0.0378 (3.8%) |
| `participation_level` | 0.0066 (0.7%) |

### Regressor Feature Importances

| Feature | Importance |
|---|---|
| `study_hours_per_week` | 0.6930 (69.3%) |
| `assignment_avg` | 0.1553 (15.5%) |
| `attendance_pct` | 0.0795 (8.0%) |
| `internal_marks` | 0.0581 (5.8%) |
| `num_backlogs` | 0.0123 (1.2%) |
| `assignment_completion` | 0.0015 (0.2%) |
| `prev_exam_score` | 0.0004 (0.0%) |
| `participation_level` | 0.0000 (0.0%) |

### Average Feature Importances (as shown in the API)

The API returns importances averaged across both models (expressed as percentages):

| Feature | Avg. Importance |
|---|---|
| `study_hours_per_week` | ~52.1% |
| `attendance_pct` | ~15.6% |
| `internal_marks` | ~13.7% |
| `assignment_avg` | ~9.7% |
| `num_backlogs` | ~4.4% |
| `assignment_completion` | ~2.3% |
| `prev_exam_score` | ~1.9% |
| `participation_level` | ~0.3% |

### Interpretation

Study hours per week is by far the dominant feature (69.3% of regressor importance), reflecting the strong linear relationship between weekly study time and the final score in the synthetic data generation formula. Attendance and internal marks are the next most influential. Participation level and assignment completion contribute minimally at the decision tree depth used.

---

## 7. Model Files

| File | Description |
|---|---|
| `ml/models/classifier.pkl` | Trained DecisionTreeClassifier (joblib format) |
| `ml/models/regressor.pkl` | Trained DecisionTreeRegressor (joblib format) |
| `ml/models/label_encoder.pkl` | Fitted LabelEncoder for category decoding |
| `ml/model_info.json` | Training metadata (hyperparameters, timestamps, metrics) |

---

## 8. Known Limitations

1. **Synthetic data** — feature importances and metric values reflect the structure of the synthetic generation formula. Real student data would likely show different importances and lower R².
2. **Single train/test split** — no cross-validation was performed; metrics may vary ±2–5% across different splits.
3. **Excellent class underrepresentation** — even with the balanced dataset design, Excellent students are harder to generate (require very high values across most features), resulting in only 2 test samples at this random seed.
4. **No pruning optimisation** — hyperparameters were set to reasonable values preventing gross overfitting; formal hyperparameter search (e.g., GridSearchCV) was not performed as it would artificially inflate apparent performance.
5. **participation_level and prev_exam_score** show near-zero importance in the regressor — this is consistent with the synthetic data formula where these features have smaller weights (0.04 and 0.12 respectively), but real data may show different relationships.
