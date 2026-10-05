"""
predict.py
----------
Reusable prediction helper used by the Flask backend.

Loads the trained classifier, regressor, and label encoder from disk,
then provides a single predict() function that accepts raw student input
and returns:
  - predicted_score        : float (regression output)
  - performance_category   : str   (classification output)
  - confidence             : dict  {category: probability}
  - feature_importances    : dict  {feature: importance}
  - recommendations        : list of str
"""

import os
import joblib
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")

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

# Lazy-load models once
_clf = None
_reg = None
_le  = None


def _load_models():
    global _clf, _reg, _le
    if _clf is None:
        clf_path = os.path.join(MODELS_DIR, "classifier.pkl")
        reg_path = os.path.join(MODELS_DIR, "regressor.pkl")
        le_path  = os.path.join(MODELS_DIR, "label_encoder.pkl")
        for p in (clf_path, reg_path, le_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Model file not found: {p}\n"
                    "Run ml/train_model.py first."
                )
        _clf = joblib.load(clf_path)
        _reg = joblib.load(reg_path)
        _le  = joblib.load(le_path)


def _generate_recommendations(features: dict, category: str) -> list:
    """Rule-based personalised recommendations derived from input features."""
    recs = []

    if features["attendance_pct"] < 75:
        recs.append(
            "📅 Attendance is below 75%. Regular class attendance "
            "significantly improves understanding and exam scores."
        )
    if features["study_hours_per_week"] < 10:
        recs.append(
            "📚 Study hours are low. Aim for at least 10–15 focused "
            "study hours per week to reinforce learning."
        )
    if features["assignment_avg"] < 60:
        recs.append(
            "✏️ Assignment average is below 60. Review feedback on past "
            "assignments and seek help from instructors."
        )
    if features["assignment_completion"] < 80:
        recs.append(
            "✅ Assignment completion is below 80%. Completing all assignments "
            "builds foundational knowledge and improves scores."
        )
    if features["participation_level"] == 0:
        recs.append(
            "🙋 Low class participation detected. Actively asking questions "
            "and joining discussions deepens conceptual understanding."
        )
    if features["num_backlogs"] > 2:
        recs.append(
            "⚠️ Multiple backlogs identified. Prioritise clearing backlogs "
            "one subject at a time with a structured revision plan."
        )
    if features["internal_marks"] < 25:
        recs.append(
            "📝 Internal marks are low. Focus on internal assessment preparation "
            "— these marks directly boost your overall performance."
        )
    if features["prev_exam_score"] < 50:
        recs.append(
            "📊 Previous exam score is below 50. Analyse past exam mistakes, "
            "practice previous question papers, and identify weak topics."
        )

    # Category-level generic advice
    if category == "Excellent":
        recs.append(
            "🌟 Excellent performance! Maintain your consistency and consider "
            "mentoring peers or exploring advanced topics."
        )
    elif category == "Good" and not recs:
        recs.append(
            "👍 Good performance! A little more consistency in weaker areas "
            "could push you to Excellent."
        )
    elif category == "Average" and not recs:
        recs.append(
            "💡 Average performance. Set specific weekly study goals and "
            "review class notes regularly to improve."
        )
    elif category == "Needs Improvement" and not recs:
        recs.append(
            "🔴 Performance needs improvement. Please speak with your academic "
            "advisor to create a targeted study plan."
        )

    return recs if recs else ["Keep up the consistent effort! 🎯"]


def predict(input_data: dict) -> dict:
    """
    Parameters
    ----------
    input_data : dict with keys matching FEATURE_COLS

    Returns
    -------
    dict with keys:
        predicted_score, performance_category, confidence,
        feature_importances, recommendations
    """
    _load_models()

    # Build feature vector in correct order
    x = np.array([[input_data[col] for col in FEATURE_COLS]])

    # Classification
    cat_encoded  = _clf.predict(x)[0]
    cat_proba    = _clf.predict_proba(x)[0]
    category     = _le.inverse_transform([cat_encoded])[0]
    confidence   = {
        str(_le.inverse_transform([i])[0]): round(float(p), 4)
        for i, p in enumerate(cat_proba)
    }

    # Regression
    score = float(_reg.predict(x)[0])
    score = round(max(0.0, min(100.0, score)), 2)

    # Feature importances (average of both models)
    clf_fi = _clf.feature_importances_
    reg_fi = _reg.feature_importances_
    avg_fi = (clf_fi + reg_fi) / 2
    feature_importances = {
        col: round(float(avg_fi[i]), 4)
        for i, col in enumerate(FEATURE_COLS)
    }

    recommendations = _generate_recommendations(input_data, category)

    return {
        "predicted_score":      score,
        "performance_category": category,
        "confidence":           confidence,
        "feature_importances":  feature_importances,
        "recommendations":      recommendations,
    }
