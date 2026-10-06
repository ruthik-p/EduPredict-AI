"""
app.py
------
EduPredict AI — Flask REST API Backend

Exposes two endpoints:
    GET  /health   → liveness check
    POST /predict  → student performance prediction

The prediction pipeline:
    1. Validate incoming JSON input
    2. Build a feature vector in the same order used during training
    3. Run DecisionTreeClassifier  → performance category + class probabilities
    4. Run DecisionTreeRegressor   → predicted numerical score
    5. Compute feature importances (average of both models)
    6. Generate rule-based personalised recommendations
    7. Return a clean JSON response

IMPORTANT: Decision Tree models do NOT require feature scaling.
           No StandardScaler is applied here or during training.

Educational note: This file is intentionally kept simple and well-commented
so that it is easy to understand and explain during an internship evaluation.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow cross-origin requests from the frontend

# ── Paths ──────────────────────────────────────────────────────────────────────
# Resolve paths relative to this file so the server can be started from any
# working directory.
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")

CLF_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
REG_PATH = os.path.join(MODELS_DIR, "regressor.pkl")
LE_PATH  = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ── Feature order ──────────────────────────────────────────────────────────────
# MUST match the column order used in ml/train_model.py exactly.
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

# ── Input validation ranges ────────────────────────────────────────────────────
# Values outside these ranges are rejected with a clear error message.
VALID_RANGES = {
    "attendance_pct":        (0,   100),
    "study_hours_per_week":  (0,   35),
    "assignment_avg":        (0,   100),
    "internal_marks":        (0,   50),
    "prev_exam_score":       (0,   100),
    "assignment_completion": (0,   100),
    "participation_level":   (0,   2),
    "num_backlogs":          (0,   5),
}

# ── Lazy-loaded models ─────────────────────────────────────────────────────────
# Models are loaded once on first request and cached in these module-level
# variables to avoid re-loading on every API call.
_clf = None
_reg = None
_le  = None


def load_models():
    """Load trained models from disk. Raises RuntimeError if files are missing."""
    global _clf, _reg, _le

    if _clf is not None:
        return  # Already loaded

    missing = [p for p in (CLF_PATH, REG_PATH, LE_PATH) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Model file(s) not found: {missing}. "
            "Please run ml/train_model.py first to generate the model files."
        )

    _clf = joblib.load(CLF_PATH)
    _reg = joblib.load(REG_PATH)
    _le  = joblib.load(LE_PATH)


# ── Input validation ───────────────────────────────────────────────────────────

def validate_input(data: dict):
    """
    Validate the incoming JSON payload.

    Returns:
        (features_dict, None)       on success
        (None, error_message_str)   on failure
    """
    errors = []

    # 1. Check all required fields are present
    for field in FEATURE_COLS:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return None, "; ".join(errors)

    # 2. Check each value is numeric and within the allowed range
    clean = {}
    for field in FEATURE_COLS:
        raw = data[field]

        # Must be a number (int or float), but NOT a boolean.
        # In Python, bool is a subclass of int, so isinstance(True, int) is True.
        # We explicitly reject booleans because JSON true/false are not valid inputs.
        if isinstance(raw, bool):
            errors.append(f"Field '{field}' must be a number, got: bool")
            continue

        if not isinstance(raw, (int, float)):
            errors.append(f"Field '{field}' must be a number, got: {type(raw).__name__}")
            continue

        # participation_level and num_backlogs must be integers
        if field in ("participation_level", "num_backlogs"):
            if not isinstance(raw, int) and not float(raw).is_integer():
                errors.append(f"Field '{field}' must be a whole number (0, 1, 2 …)")
                continue
            raw = int(raw)

        lo, hi = VALID_RANGES[field]
        if not (lo <= raw <= hi):
            errors.append(
                f"Field '{field}' = {raw} is out of range [{lo}, {hi}]"
            )
            continue

        clean[field] = float(raw)

    if errors:
        return None, "; ".join(errors)

    return clean, None


# ── Recommendation engine ──────────────────────────────────────────────────────

def generate_recommendations(features: dict, category: str) -> list:
    """
    Rule-based personalised recommendations derived from the student's input.
    Each rule checks a specific feature threshold and adds a targeted suggestion.
    """
    recs = []

    if features["attendance_pct"] < 75:
        recs.append(
            "Attendance is below 75%. Regular attendance is one of the strongest "
            "predictors of academic success — try not to miss classes."
        )

    if features["study_hours_per_week"] < 10:
        recs.append(
            "Study hours are low (under 10 hrs/week). Aim for at least 10–15 focused "
            "hours per week — use a timetable to schedule dedicated study blocks."
        )

    if features["assignment_avg"] < 60:
        recs.append(
            "Assignment average is below 60. Review instructor feedback carefully on "
            "past assignments and visit office hours to clarify weak areas."
        )

    if features["assignment_completion"] < 80:
        recs.append(
            "Assignment completion rate is below 80%. Submitting all assignments, even "
            "imperfectly, builds knowledge and contributes significantly to your grade."
        )

    if features["internal_marks"] < 25:
        recs.append(
            "Internal marks are below 50% of maximum (25/50). Prepare thoroughly for "
            "internal tests — revise lecture notes and attempt past question papers."
        )

    if features["prev_exam_score"] < 50:
        recs.append(
            "Previous exam score is below 50. Analyse where marks were lost, identify "
            "weak topics, and practise solving past exam questions under timed conditions."
        )

    if int(features["participation_level"]) == 0:
        recs.append(
            "Class participation is recorded as Low. Actively asking questions and "
            "joining discussions deepens understanding and keeps you engaged."
        )

    if int(features["num_backlogs"]) > 0:
        recs.append(
            f"You have {int(features['num_backlogs'])} backlog(s). Prioritise clearing "
            "them one subject at a time — unresolved backlogs create compounding difficulty."
        )

    # Category-level advice when no specific red flags were found
    if not recs:
        if category == "Excellent":
            recs.append(
                "Outstanding performance! Maintain your consistency and consider "
                "taking on leadership roles, tutoring peers, or exploring advanced topics."
            )
        elif category == "Good":
            recs.append(
                "Good performance. A focused push in your weaker areas — particularly "
                "improving study consistency — could move you to Excellent."
            )
        elif category == "Average":
            recs.append(
                "Average performance. Set specific weekly targets for study hours and "
                "assignment completion to build steady improvement."
            )
        else:
            recs.append(
                "Performance needs improvement. Consider speaking with your academic "
                "advisor to create a structured, personalised study plan."
            )

    return recs


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """
    GET /health
    Liveness check — confirms the API server is running.
    """
    return jsonify({
        "status":  "healthy",
        "service": "EduPredict AI",
        "version": "1.0.0",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts a JSON body with student feature values.
    Returns predicted score, category, confidence, feature importance,
    and personalised recommendations.
    """
    # ── Step 1: Parse JSON body ────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({
            "error": "Request body must be JSON. "
                     "Set Content-Type: application/json header."
        }), 400

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON in request body."}), 400

    if data is None:
        return jsonify({"error": "Request body is empty or not valid JSON."}), 400

    # ── Step 2: Validate inputs ────────────────────────────────────────────────
    features, err = validate_input(data)
    if err:
        return jsonify({"error": err}), 422

    # ── Step 3: Load models (first request only) ───────────────────────────────
    try:
        load_models()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    # ── Step 4: Build feature vector in training order ─────────────────────────
    # Use a DataFrame with named columns to match how the model was trained and
    # to suppress sklearn's feature-name warning.
    x = pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)

    # ── Step 5: Classification ─────────────────────────────────────────────────
    try:
        cat_encoded = _clf.predict(x)[0]
        cat_proba   = _clf.predict_proba(x)[0]
        category    = _le.inverse_transform([cat_encoded])[0]
        confidence  = {
            str(_le.inverse_transform([i])[0]): round(float(p) * 100, 1)
            for i, p in enumerate(cat_proba)
        }  # Expressed as percentages for easy display
    except Exception as e:
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500

    # ── Step 6: Regression ─────────────────────────────────────────────────────
    try:
        raw_score      = float(_reg.predict(x)[0])
        predicted_score = round(max(0.0, min(100.0, raw_score)), 2)
    except Exception as e:
        return jsonify({"error": f"Regression prediction failed: {str(e)}"}), 500

    # ── Step 7: Feature importances (average of classifier + regressor) ────────
    clf_fi = _clf.feature_importances_
    reg_fi = _reg.feature_importances_
    avg_fi = (clf_fi + reg_fi) / 2.0
    feature_importance = {
        col: round(float(avg_fi[i]) * 100, 2)   # as percentage
        for i, col in enumerate(FEATURE_COLS)
    }
    # Sort descending so the frontend can display top factors easily
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    )

    # ── Step 8: Personalised recommendations ──────────────────────────────────
    recommendations = generate_recommendations(features, category)

    # ── Step 9: Build and return response ────────────────────────────────────
    return jsonify({
        "predicted_score":           predicted_score,
        "performance_category":      category,
        "classification_confidence": confidence,
        "feature_importance":        feature_importance,
        "personalized_recommendations": recommendations,
    }), 200


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. Available: GET /health, POST /predict"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": f"Method not allowed on this endpoint."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "An unexpected server error occurred."}), 500


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  EduPredict AI — Flask API Server")
    print("=" * 55)
    print(f"  Models directory : {MODELS_DIR}")
    print(f"  Endpoints        : GET /health  |  POST /predict")
    print(f"  Starting server  : http://127.0.0.1:5000")
    print("=" * 55)

    # Pre-load models at startup so the first /predict call is fast
    try:
        load_models()
        print("  Models loaded successfully.")
    except RuntimeError as e:
        print(f"  [WARNING] {e}")
        print("  Server will start but /predict will return 503 until models exist.")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
