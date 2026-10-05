# EduPredict AI
## Intelligent Student Performance Prediction & Personalized Improvement System

> **IBM SkillsBuild + AICTE 6-Week Machine Learning & Applied AI Internship Project**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Key Features](#key-features)
5. [Technology Stack](#technology-stack)
6. [Project Architecture](#project-architecture)
7. [Project Structure](#project-structure)
8. [Dataset Description](#dataset-description)
9. [Input Features](#input-features)
10. [Data Preprocessing](#data-preprocessing)
11. [Machine Learning Algorithms](#machine-learning-algorithms)
12. [Model Evaluation Results](#model-evaluation-results)
13. [Backend API](#backend-api)
14. [Frontend](#frontend)
15. [How to Run the Project](#how-to-run-the-project)
16. [Example Prediction](#example-prediction)
17. [Limitations](#limitations)
18. [Future Enhancements](#future-enhancements)
19. [Educational Disclaimer](#educational-disclaimer)

---

## Project Overview

**EduPredict AI** is a web-based machine learning application that predicts a student's academic performance category and estimated exam score based on their academic and behavioural inputs. It uses supervised learning models trained on a synthetic dataset to demonstrate key ML concepts including classification, regression, decision trees, data preprocessing, and applied AI.

The system takes eight student input features, processes them through two trained scikit-learn Decision Tree models, and returns:
- A predicted numerical score (0–100)
- A performance category (Excellent / Good / Average / Needs Improvement)
- Classification confidence probabilities
- Feature importance rankings
- Personalised improvement recommendations

---

## Problem Statement

Students and educators often lack early, data-driven insight into academic performance trends. Identifying at-risk students early and providing targeted guidance can significantly improve outcomes. This project demonstrates how supervised machine learning can be applied to predict student performance and generate personalised improvement plans.

---

## Objectives

1. Build a supervised learning pipeline using scikit-learn Decision Trees.
2. Demonstrate both classification and regression in a single application.
3. Implement data preprocessing best practices (missing-value handling, duplicate detection, range validation, train/test split without leakage).
4. Evaluate models using standard classification and regression metrics.
5. Deploy the ML pipeline through a Flask REST API.
6. Present results through a clean, responsive web frontend.
7. Generate rule-based personalised recommendations from model output.

---

## Key Features

- **Dual ML models** — DecisionTreeClassifier for category prediction + DecisionTreeRegressor for score prediction
- **8 academic input features** with full validation
- **Classification confidence** — probability for each of the 4 categories
- **Feature importance** — which inputs most influenced the prediction
- **Personalised recommendations** — targeted advice based on weak areas
- **Animated score ring** — visual score display
- **Professional responsive UI** — works on desktop, tablet, and mobile
- **REST API** — clean JSON endpoints with full error handling
- **CORS enabled** — frontend can call API from any origin

---

## Technology Stack

| Layer | Technology |
|---|---|
| Machine Learning | Python 3, scikit-learn (DecisionTreeClassifier, DecisionTreeRegressor) |
| Data Processing | pandas, numpy |
| Model Persistence | joblib |
| Backend API | Flask, flask-cors |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Testing | Python unittest |

---

## Project Architecture

```
Browser (HTML/CSS/JS)
        │
        │  POST /predict  (JSON)
        ▼
Flask REST API  (backend/app.py)
        │
        ├── Input Validation
        │
        ├── DecisionTreeClassifier ──► Performance Category + Confidence
        │
        ├── DecisionTreeRegressor  ──► Predicted Score
        │
        ├── Feature Importance (avg of both models)
        │
        └── Recommendation Engine ──► Personalised Suggestions
                                          │
                                          ▼
                              JSON Response to Browser
```

---

## Project Structure

```
EduPredict-AI/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator (500 records)
│   └── student_data.csv        # Generated synthetic dataset
│
├── ml/
│   ├── train_model.py          # Training pipeline + evaluation + saves models
│   ├── predict.py              # Reusable prediction helper
│   └── models/
│       ├── classifier.pkl      # Trained DecisionTreeClassifier
│       ├── regressor.pkl       # Trained DecisionTreeRegressor
│       └── label_encoder.pkl   # LabelEncoder for category labels
│
├── backend/
│   ├── app.py                  # Flask REST API
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── index.html              # Main UI page
│   ├── style.css               # Stylesheet
│   └── script.js               # API calls + result rendering
│
├── tests/
│   └── test_predictions.py     # Permanent unittest suite
│
├── documentation/
│   ├── PROJECT_REPORT.md       # Formal internship project report
│   └── MODEL_EVALUATION.md     # Detailed ML evaluation results
│
└── README.md                   # This file
```

---

## Dataset Description

> **SYNTHETIC DATA NOTICE:** The dataset used in this project is entirely programmatically generated. It contains **no real student names, IDs, roll numbers, or personal information** of any kind. It was created solely for the purpose of demonstrating machine learning concepts in an educational setting.

| Property | Value |
|---|---|
| Total records | 500 |
| Generation method | Profile-based synthetic generation with weighted score formula + noise |
| Random seed | 42 (fully reproducible) |
| Missing values | 0 |
| Duplicate records | 0 |
| Features | 8 |
| Target (classification) | `performance_category` (4 classes) |
| Target (regression) | `final_score` (continuous, 0–100) |

**Class distribution:**

| Category | Count | Percentage |
|---|---|---|
| Excellent | ~96 | ~19.2% |
| Good | ~146 | ~29.2% |
| Average | ~139 | ~27.8% |
| Needs Improvement | ~119 | ~23.8% |

To regenerate the dataset:
```bash
python data/generate_dataset.py
```

---

## Input Features

| Feature | Description | Range |
|---|---|---|
| `attendance_pct` | Percentage of classes attended | 0 – 100 |
| `study_hours_per_week` | Average hours studied per week | 0 – 35 |
| `assignment_avg` | Average assignment score | 0 – 100 |
| `internal_marks` | Internal assessment marks | 0 – 50 |
| `prev_exam_score` | Score in the most recent previous exam | 0 – 100 |
| `assignment_completion` | Percentage of assignments submitted | 0 – 100 |
| `participation_level` | Class participation (0=Low, 1=Medium, 2=High) | 0, 1, 2 |
| `num_backlogs` | Number of subjects with pending/failed status | 0 – 5 |

---

## Data Preprocessing

The following preprocessing steps are applied in `ml/train_model.py` **before** the train/test split to prevent data leakage:

1. **Missing value detection** — rows with any null values are dropped (none in synthetic data)
2. **Duplicate detection** — duplicate rows are removed (none in synthetic data)
3. **Range validation** — all feature values are checked against allowed bounds; out-of-range values are clipped with a warning
4. **Categorical encoding** — `performance_category` labels are encoded using `sklearn.preprocessing.LabelEncoder` with fixed class order
5. **Train/test split** — 80% train / 20% test using `train_test_split` with `stratify=y` to preserve class proportions
6. **No feature scaling** — Decision Trees are invariant to feature scale; no StandardScaler is applied

---

## Machine Learning Algorithms

### Decision Tree Classifier

**Purpose:** Predict the performance category (Excellent / Good / Average / Needs Improvement).

A Decision Tree Classifier learns a tree of if-else rules from training data. Each internal node tests a feature value, each branch represents an outcome, and each leaf node holds a class label. This algorithm is:
- Highly interpretable — the learned rules can be visualised and explained
- Scale-invariant — no normalisation required
- Capable of multi-class classification natively

**Hyperparameters used:**

| Parameter | Value | Reason |
|---|---|---|
| `max_depth` | 6 | Prevents overfitting by limiting tree depth |
| `min_samples_split` | 10 | Requires at least 10 samples to split a node |
| `min_samples_leaf` | 5 | Requires at least 5 samples in any leaf |
| `random_state` | 42 | Reproducibility |

### Decision Tree Regressor

**Purpose:** Predict the numerical exam score (0–100 continuous).

Uses the same tree-splitting approach as the classifier, but leaf nodes contain mean target values instead of class labels. Produces an interpretable regression model with feature importances.

**Hyperparameters:** identical to the classifier (`max_depth=6`, `min_samples_split=10`, `min_samples_leaf=5`, `random_state=42`).

---

## Model Evaluation Results

> Results are from the held-out **test set (20%, 100 samples)**. These are genuine test-set results — not training-set results.

### Classification — DecisionTreeClassifier

| Metric | Score |
|---|---|
| **Accuracy** | **76.00%** |
| Weighted Precision | 77.80% |
| Weighted Recall | 76.00% |
| Weighted F1-score | 75.98% |

### Regression — DecisionTreeRegressor

| Metric | Value |
|---|---|
| **MAE** | **4.0588** |
| MSE | 26.8795 |
| **RMSE** | **5.1845** |
| **R²** | **0.9129** |

An R² of 0.91 means the regressor explains **91.3%** of the variance in student scores on the test set.

For detailed per-class metrics, confusion matrix, and feature importances, see [`documentation/MODEL_EVALUATION.md`](documentation/MODEL_EVALUATION.md).

---

## Backend API

**Start the server:**
```bash
cd backend
python app.py
# Server runs at http://127.0.0.1:5000
```

### Endpoints

#### `GET /health`
Returns server liveness status.

```json
{ "status": "healthy", "service": "EduPredict AI", "version": "1.0.0" }
```

#### `POST /predict`
Accepts a JSON body with 8 student features. Returns full prediction results.

**Error responses:**
- `400` — malformed or missing JSON body
- `422` — validation error (missing field, out of range, wrong type)
- `503` — model files not found (run `ml/train_model.py` first)

See [Example Prediction](#example-prediction) below.

---

## Frontend

Open `frontend/index.html` directly in any browser (no build step required).

**Features:**
- Input form with labels, hints, and min/max constraints
- Client-side validation before any API call
- Loading spinner while waiting for response
- Animated SVG score ring (colour changes by category)
- Category badge (colour-coded: green/blue/amber/red)
- Confidence bars for all 4 categories
- Feature importance bar chart (pure CSS/JS — no charting library)
- Personalised recommendation cards
- Error display if server is unreachable or returns an error
- Reset / clear button
- Fully responsive (desktop → tablet → mobile)

---

## How to Run the Project

### Prerequisites

- Python 3.9 or higher
- pip

### Step 1 — Install dependencies

```bash
cd EduPredict-AI/backend
pip install -r requirements.txt
```

### Step 2 — Generate the dataset (already included, but can be regenerated)

```bash
python data/generate_dataset.py
```

### Step 3 — Train the models (already included, but can be retrained)

```bash
python ml/train_model.py
```

### Step 4 — Start the Flask backend

```bash
cd backend
python app.py
```

### Step 5 — Open the frontend

Open `frontend/index.html` in any modern web browser.

### Step 6 — Run the test suite

```bash
cd EduPredict-AI
python -m pytest tests/test_predictions.py -v
# or
python -m unittest discover tests
```

---

## Example Prediction

**Request:**
```json
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "attendance_pct": 85,
  "study_hours_per_week": 20,
  "assignment_avg": 78,
  "internal_marks": 38,
  "prev_exam_score": 72,
  "assignment_completion": 90,
  "participation_level": 2,
  "num_backlogs": 0
}
```

**Response:**
```json
{
  "predicted_score": 69.52,
  "performance_category": "Good",
  "classification_confidence": {
    "Average": 12.5,
    "Excellent": 0.0,
    "Good": 87.5,
    "Needs Improvement": 0.0
  },
  "feature_importance": {
    "study_hours_per_week": 52.05,
    "attendance_pct": 15.6,
    "internal_marks": 13.72,
    "assignment_avg": 9.67,
    "num_backlogs": 4.41,
    "assignment_completion": 2.31,
    "prev_exam_score": 1.91,
    "participation_level": 0.33
  },
  "personalized_recommendations": [
    "Good performance. A focused push in your weaker areas could move you to Excellent."
  ]
}
```

---

## Limitations

1. **Synthetic dataset** — models are trained on programmatically generated data, not real student records. Predictions should not be used for actual academic decisions.
2. **Small dataset** — 500 records with 80/20 split leaves only 100 test samples; metrics may vary slightly with different seeds.
3. **No temporal features** — the model does not account for improvement over time.
4. **Rule-based recommendations** — suggestions are threshold-based, not learned from data.
5. **No authentication** — the API has no access control; not suitable for production deployment as-is.
6. **Local deployment only** — the frontend hardcodes `http://127.0.0.1:5000`; would need configuration for remote deployment.

---

## Future Enhancements

- Train on real (anonymised, consented) student data to improve generalisability
- Add Random Forest or Gradient Boosting models for comparison
- Implement cross-validation for more robust metric estimates
- Add a student history feature to track improvement over time
- Support multiple subject predictions
- Add authentication and a database backend
- Deploy to a cloud platform (IBM Cloud, Heroku, etc.)
- Internationalise the UI for multiple languages

---

## Educational Disclaimer

> **EduPredict AI is an educational demonstration project created as part of the IBM SkillsBuild + AICTE 6-Week Machine Learning & Applied AI Internship.**
>
> - All predictions are based on a **synthetic (artificially generated) dataset** that contains **no real student personal information**.
> - This system **should not be used as an official academic assessment tool** or to make real decisions about students.
> - Results are for demonstration and learning purposes only.
> - The project is designed to illustrate supervised learning, classification, regression, and applied AI concepts in an accessible way.
