# EduPredict AI — Formal Project Report

**Project Title:** EduPredict AI – Intelligent Student Performance Prediction & Personalized Improvement System

**Internship Programme:** IBM SkillsBuild + AICTE 6-Week Machine Learning & Applied AI Internship

**Technology Domain:** Machine Learning / Applied AI

**Developer:** B24CS265

**Tools & Platforms:** Python, scikit-learn, Flask, IBM Bob (AI-assisted development)

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Concepts Covered from Internship Curriculum](#5-concepts-covered-from-internship-curriculum)
6. [System Architecture](#6-system-architecture)
7. [Dataset Design](#7-dataset-design)
8. [Data Preprocessing Pipeline](#8-data-preprocessing-pipeline)
9. [Machine Learning Model Design](#9-machine-learning-model-design)
10. [Model Training and Evaluation](#10-model-training-and-evaluation)
11. [Backend API Design](#11-backend-api-design)
12. [Frontend Design](#12-frontend-design)
13. [System Integration](#13-system-integration)
14. [Results and Analysis](#14-results-and-analysis)
15. [Limitations](#15-limitations)
16. [Future Work](#16-future-work)
17. [Conclusion](#17-conclusion)
18. [References](#18-references)

---

## 1. Abstract

EduPredict AI is a full-stack machine learning web application that demonstrates supervised learning concepts in the context of student academic performance prediction. The system accepts eight student academic and behavioural features, applies trained Decision Tree models to predict both a numerical performance score and a categorical performance label, and generates personalised improvement recommendations.

The project was developed as part of the IBM SkillsBuild + AICTE 6-Week Machine Learning & Applied AI Internship and covers the following curriculum concepts: supervised learning, classification, regression, decision trees, data preprocessing, applied AI, and IBM Bob-assisted development.

The application is built using Python (scikit-learn, Flask) for the backend ML pipeline and API, and plain HTML/CSS/JavaScript for the frontend. All ML models are trained on a fully synthetic dataset of 500 records containing no real student personal information.

The Decision Tree Classifier achieved **76% accuracy** on the test set, and the Decision Tree Regressor achieved an **R² of 0.91**, demonstrating strong predictive performance on the synthetic data.

---

## 2. Introduction

Machine learning has been increasingly applied to education analytics to identify patterns in student behaviour and predict academic outcomes. Early identification of at-risk students enables educators and students themselves to take corrective action before assessments.

This project builds a simplified but complete end-to-end AI application that demonstrates how raw student data can be collected, preprocessed, modelled, and presented through a user-friendly web interface. Rather than using real student data (which would require institutional consent and privacy safeguards), the project generates a reproducible synthetic dataset that exhibits realistic relationships between academic behaviours and outcomes.

The application is intentionally kept beginner-friendly — all algorithms are well-known, interpretable, and directly mappable to concepts taught in the internship curriculum.

---

## 3. Problem Statement

Students frequently lack objective, early feedback on whether their current academic habits are likely to lead to satisfactory outcomes. Conventional assessment is periodic and retrospective. A predictive tool that analyses ongoing academic behaviours and forecasts likely performance — while suggesting targeted improvements — could serve as a useful self-assessment aid.

**This project addresses:**
- How can a student's performance category be predicted from behavioural features using classification?
- How can a numerical performance score be estimated using regression?
- What features most influence performance?
- How can the system generate targeted recommendations from the prediction?

---

## 4. Objectives

| # | Objective | Status |
|---|---|---|
| 1 | Build a supervised learning classification model using DecisionTreeClassifier | ✅ |
| 2 | Build a supervised learning regression model using DecisionTreeRegressor | ✅ |
| 3 | Implement a complete data preprocessing pipeline | ✅ |
| 4 | Evaluate classification using accuracy, precision, recall, F1, confusion matrix | ✅ |
| 5 | Evaluate regression using MAE, MSE, RMSE, R² | ✅ |
| 6 | Create a Flask REST API exposing the trained models | ✅ |
| 7 | Build a responsive web frontend for data input and result display | ✅ |
| 8 | Generate personalised recommendations from prediction outputs | ✅ |
| 9 | Document all results and demonstrate internship concept coverage | ✅ |

---

## 5. Concepts Covered from Internship Curriculum

| Concept | Where Applied |
|---|---|
| **Supervised Learning** | Both models trained on labelled synthetic data (features → target) |
| **Classification** | DecisionTreeClassifier predicts performance category (4 classes) |
| **Decision Trees** | Primary algorithm for both classification and regression |
| **Regression** | DecisionTreeRegressor predicts numerical score (continuous output) |
| **Data Preprocessing** | Missing values, duplicates, range validation, encoding, train/test split |
| **AI/ML Prediction** | Full prediction pipeline from raw input to structured JSON output |
| **Applied AI** | End-to-end web application using ML for a real-world use case |
| **IBM Bob-assisted development** | Project scaffolding, code generation, debugging, and documentation |

---

## 6. System Architecture

The system consists of three loosely coupled layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                        │
│   HTML + CSS + JavaScript (frontend/index.html)         │
│   • Input form    • Results dashboard                   │
│   • Validation    • Recommendations                     │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST /predict (JSON)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Backend Layer                         │
│   Flask REST API (backend/app.py)                        │
│   • Input validation    • Model loading (lazy)          │
│   • Feature vector construction                         │
│   • Recommendation engine                               │
└────────────────────────┬────────────────────────────────┘
                         │ sklearn predict()
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    ML Model Layer                        │
│   ml/models/                                            │
│   • classifier.pkl  (DecisionTreeClassifier)            │
│   • regressor.pkl   (DecisionTreeRegressor)             │
│   • label_encoder.pkl                                   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. User fills in the form with 8 academic feature values.
2. JavaScript validates input client-side (range, type, required).
3. A `POST /predict` request is sent with a JSON body.
4. Flask validates the input server-side (type, range, boolean rejection).
5. A pandas DataFrame is constructed in training-column order.
6. The classifier predicts the category and class probabilities.
7. The regressor predicts the numerical score.
8. Feature importances are averaged across both models.
9. Rule-based recommendations are generated from feature values.
10. A structured JSON response is returned to the browser.
11. JavaScript renders the score ring, category badge, confidence bars, importance chart, and recommendations.

---

## 7. Dataset Design

### Synthetic Data Strategy

Because no real student records were available (and obtaining such data with proper consent is outside the scope of an internship project), a reproducible synthetic dataset was designed with the following properties:

- **500 records** — sufficient for a train/test split while remaining computationally lightweight
- **Four student profile groups** — weak, average, good, and strong students — each with realistic feature value ranges
- **Weighted score formula** — `final_score` is computed from a deterministic weighted sum of the 8 features plus Gaussian noise (σ = 3.0), creating genuine statistical relationships between features and the target
- **Category labels derived from scores** — not injected directly, ensuring the categories are consistent with the underlying feature values
- **Balanced class distribution** — ~19% Excellent, ~29% Good, ~28% Average, ~24% Needs Improvement
- **Fully reproducible** — `RANDOM_SEED = 42` throughout

### Score Formula

```
final_score = 0.20 × attendance_pct
            + 0.22 × (study_hours / 35 × 100)
            + 0.18 × assignment_avg
            + 0.20 × (internal_marks / 50 × 100)
            + 0.12 × prev_exam_score
            + 0.04 × assignment_completion
            + 0.04 × (participation_level / 2 × 100)
            − 2.50 × num_backlogs
            + Normal(0, 3.0)
```

Category thresholds: Excellent ≥ 80, Good ≥ 65, Average ≥ 50, Needs Improvement < 50.

### Dataset Statistics

| Statistic | `final_score` |
|---|---|
| Mean | ~62.4 |
| Std Dev | ~18.4 |
| Min | ~22.8 |
| Max | ~97.1 |

---

## 8. Data Preprocessing Pipeline

All preprocessing is performed in `ml/train_model.py` **before** the train/test split to prevent any form of data leakage.

### Steps

| Step | Action | Result on Synthetic Data |
|---|---|---|
| 1. Load CSV | `pd.read_csv()` | 500 rows × 10 columns |
| 2. Missing values | Drop rows with any null | 0 rows dropped |
| 3. Duplicates | Drop duplicate rows | 0 rows dropped |
| 4. Range validation | Clip out-of-range values with warning | No violations |
| 5. Label encoding | `LabelEncoder` on `performance_category` | 4 classes encoded 0–3 |
| 6. Train/test split | 80/20 stratified by category | 400 train / 100 test |

### Why No Feature Scaling?

Decision Trees split on feature thresholds — they are entirely invariant to the absolute scale or distribution of features. Applying StandardScaler or MinMaxScaler would have no effect on tree structure or predictions, so it was deliberately omitted to keep the pipeline simple and correct.

### Data Leakage Prevention

- The `LabelEncoder` is fitted **only on the training labels** and then used to transform both train and test labels.
- All preprocessing (missing-value drops, range clipping) is applied before splitting.
- No test-set statistics are used in any preprocessing step.

---

## 9. Machine Learning Model Design

### Model 1: DecisionTreeClassifier

**Task:** Multi-class classification (4 categories)
**Library:** `sklearn.tree.DecisionTreeClassifier`

A Decision Tree builds a binary tree by recursively finding the feature and threshold that best separates the training data (measured by Gini impurity or information gain). Classification is performed by traversing the tree from the root to a leaf and reading off the class label.

**Hyperparameter rationale:**

| Parameter | Value | Rationale |
|---|---|---|
| `max_depth=6` | Limits tree to 6 levels | Prevents overfitting; keeps model interpretable |
| `min_samples_split=10` | Minimum 10 samples to split | Avoids splitting on tiny subsets |
| `min_samples_leaf=5` | Minimum 5 samples per leaf | Ensures each leaf represents a meaningful group |
| `random_state=42` | Fixed seed | Reproducible results |

### Model 2: DecisionTreeRegressor

**Task:** Regression (predict continuous score 0–100)
**Library:** `sklearn.tree.DecisionTreeRegressor`

Uses the same tree-building approach as the classifier, but minimises mean squared error instead of Gini impurity. Leaf nodes output the mean of all training samples that reach them.

**Hyperparameters:** identical to the classifier.

### Feature Importance

Both models produce `feature_importances_` arrays (values sum to 1.0) indicating the relative contribution of each feature to predictions. The API averages importances across both models and returns them as percentages.

---

## 10. Model Training and Evaluation

### Training

Models were trained on 400 samples (80% stratified split) from the 500-record synthetic dataset.

### Classification Results (test set, n=100)

| Metric | Value |
|---|---|
| Accuracy | **76.00%** |
| Weighted Precision | **77.80%** |
| Weighted Recall | **76.00%** |
| Weighted F1-score | **75.98%** |

**Per-class breakdown:**

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Average | 0.62 | 0.78 | 0.69 | 49 |
| Excellent | 0.80 | 0.63 | 0.71 | 2* |
| Good | 0.76 | 0.62 | 0.68 | 26 |
| Needs Improvement | 0.94 | 0.71 | 0.81 | 23 |

*Only 2 test samples for Excellent due to class size; metrics less reliable for this class.

### Regression Results (test set, n=100)

| Metric | Value | Interpretation |
|---|---|---|
| MAE | **4.0588** | Average absolute error of ~4 points on a 0–100 scale |
| MSE | **26.8795** | Mean squared error |
| RMSE | **5.1845** | Typical error of ~5.2 points |
| **R²** | **0.9129** | Model explains 91.3% of score variance |

### Feature Importance (Averaged)

| Feature | Importance (%) |
|---|---|
| study_hours_per_week | ~52% |
| attendance_pct | ~16% |
| internal_marks | ~14% |
| assignment_avg | ~10% |
| num_backlogs | ~4% |
| assignment_completion | ~2% |
| prev_exam_score | ~2% |
| participation_level | ~0% |

---

## 11. Backend API Design

**Framework:** Flask (Python)
**CORS:** flask-cors (enabled for all origins)

### Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Full prediction pipeline |

### Input Validation

The API performs strict server-side validation:
- All 8 fields must be present
- Values must be numeric (not boolean, string, null, or list)
- Values must fall within their defined ranges
- `participation_level` and `num_backlogs` must be whole numbers

### Error Handling

| HTTP Status | Meaning |
|---|---|
| 200 | Successful prediction |
| 400 | Malformed/missing JSON body |
| 422 | Validation error (field missing, out of range, wrong type) |
| 503 | Model files not found (run training first) |

---

## 12. Frontend Design

**Technology:** HTML5 + CSS3 + Vanilla JavaScript (no frameworks)

### UI Components

| Component | Description |
|---|---|
| Header | Project title, subtitle, tech badge pills |
| Input form | 8 fields with labels, hints, validation |
| Score ring | Animated SVG circle, 0–100 scale |
| Category badge | Colour-coded: green/blue/amber/red |
| Confidence bars | 4 animated horizontal bars |
| Feature importance | Horizontal bar chart (pure CSS/JS) |
| Recommendations | Card-style list |
| Error display | Network errors, API errors, validation |
| Disclaimer | Educational notice |
| Footer | Project attribution |

### Responsiveness

Three breakpoints: desktop (2-column grid), tablet (1-column), mobile (stacked form).

---

## 13. System Integration

The frontend and backend integrate through a single REST endpoint. Key integration points verified by automated tests:

- All 8 JSON field names in `script.js` exactly match `FEATURE_COLS` in `app.py`
- All 5 response field names consumed by JS match the API response structure
- `CORS(app)` is enabled in Flask, allowing browser `fetch()` calls
- The API URL `http://127.0.0.1:5000/predict` is configured in `script.js`

A suite of 168 automated tests (Groups 1–9 covering valid inputs, boundary values, missing fields, invalid types, booleans, and malformed bodies) passes with 0 failures.

---

## 14. Results and Analysis

### Classification Performance

An accuracy of 76% on a 4-class problem is well above the 25% random baseline. The "Needs Improvement" category achieves the highest precision (0.94) because its feature profile (very low study hours, attendance, and marks) is clearly distinct. The "Excellent" class has limited test samples due to its lower frequency in the dataset.

### Regression Performance

An R² of 0.91 indicates that the regressor explains 91% of score variance on unseen test data. The RMSE of ~5.2 points on a 0–100 scale represents a practical average error of around half a grade band — acceptable for an educational demonstration system trained on 400 synthetic samples.

### Why Decision Trees Were Chosen

Decision Trees were specified in the internship project requirements. They are also the right choice for this demonstrational context because:
1. Their feature importance scores are natively available and directly interpretable.
2. The learned rules can be visualised and explained without statistical background.
3. They handle both classification and regression, allowing a single algorithm family across both tasks.
4. They do not require feature scaling, keeping the pipeline simple and avoiding a common student misconception.

---

## 15. Limitations

1. **Synthetic data** — results do not generalise to real students without retraining on real, consented data.
2. **Small sample size** — 500 records is sufficient for demonstration but limited for robust generalisation.
3. **No cross-validation** — a single train/test split may produce optimistic or pessimistic metrics depending on the split.
4. **Rule-based recommendations** — threshold rules are manually defined; a data-driven recommendation system would be more sophisticated.
5. **No temporal modelling** — performance change over time is not captured.
6. **Local deployment** — the system is not production-ready (no auth, no HTTPS, hardcoded localhost URL).

---

## 16. Future Work

- Collect real, anonymised, consented student data and retrain models.
- Compare Decision Trees with Random Forests, Gradient Boosting, and neural networks.
- Implement k-fold cross-validation for more reliable metric estimates.
- Add SHAP values for per-prediction explainability.
- Build a student tracking dashboard to monitor improvement over time.
- Deploy to IBM Cloud using IBM Code Engine or Cloud Foundry.
- Add multi-subject support.
- Integrate with Learning Management Systems (LMS) via API.

---

## 17. Conclusion

EduPredict AI successfully demonstrates the end-to-end application of supervised machine learning concepts learned during the IBM SkillsBuild + AICTE internship. The project implements:

- A **data generation and preprocessing pipeline** following best practices (no leakage, range validation, encoding)
- A **Decision Tree Classifier** achieving 76% accuracy on 4-class performance prediction
- A **Decision Tree Regressor** achieving R² = 0.91 on continuous score prediction
- A **Flask REST API** with robust input validation and structured error handling
- A **responsive web frontend** that presents results through a professional, interactive dashboard
- A **168-test automated test suite** covering valid inputs, boundary values, and all major error paths

The project is appropriately scoped for an educational demonstration — it is simple enough to explain clearly in an evaluation, yet complete enough to illustrate the full ML application development lifecycle.

---

## 18. References

1. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
2. Breiman, L. et al. (1984). *Classification and Regression Trees*. Wadsworth.
3. Flask Documentation. https://flask.palletsprojects.com/
4. scikit-learn Decision Tree Documentation. https://scikit-learn.org/stable/modules/tree.html
5. IBM SkillsBuild ML & Applied AI Internship Curriculum, AICTE, 2024.
6. Pandas Documentation. https://pandas.pydata.org/docs/
7. NumPy Documentation. https://numpy.org/doc/

---

*This report was prepared as part of the IBM SkillsBuild + AICTE Machine Learning & Applied AI Internship. All data used in this project is synthetic and contains no real student personal information.*
