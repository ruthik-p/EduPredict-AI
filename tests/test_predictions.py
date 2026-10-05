"""
tests/test_predictions.py
--------------------------
EduPredict AI — Permanent Automated Test Suite

Uses Python's built-in unittest framework with Flask's test_client so no
live server is required. Tests are self-contained and can be re-run any
time the backend or models are updated.

Run with:
    python -m unittest tests/test_predictions.py -v
    python -m pytest tests/test_predictions.py -v

Test classes:
    TestHealthEndpoint         — GET /health liveness checks
    TestPredictValidInput      — Valid predictions: named profiles + boundary values
    TestPredictValidation      — All invalid input scenarios (422 / 400 responses)
    TestResponseStructure      — Schema, types, ranges, confidence sums, feature counts
"""

import sys
import os
import unittest

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from the project root or from tests/
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_BACKEND = os.path.join(_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import app as _app

# Point model paths to ml/models/ regardless of cwd
_app.BASE_DIR   = _ROOT
_app.MODELS_DIR = os.path.join(_ROOT, "ml", "models")
_app.CLF_PATH   = os.path.join(_app.MODELS_DIR, "classifier.pkl")
_app.REG_PATH   = os.path.join(_app.MODELS_DIR, "regressor.pkl")
_app.LE_PATH    = os.path.join(_app.MODELS_DIR, "label_encoder.pkl")
# Reset lazy-loaded model cache so each test run reloads cleanly
_app._clf = _app._reg = _app._le = None

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_CATEGORIES = {"Excellent", "Good", "Average", "Needs Improvement"}
FEATURE_COLS     = [
    "attendance_pct", "study_hours_per_week", "assignment_avg",
    "internal_marks", "prev_exam_score", "assignment_completion",
    "participation_level", "num_backlogs",
]
REQUIRED_RESPONSE_KEYS = [
    "predicted_score", "performance_category", "classification_confidence",
    "feature_importance", "personalized_recommendations",
]

# ── Named test payloads ───────────────────────────────────────────────────────
STRONG_STUDENT = {
    "attendance_pct": 95, "study_hours_per_week": 30, "assignment_avg": 90,
    "internal_marks": 45, "prev_exam_score": 88, "assignment_completion": 98,
    "participation_level": 2, "num_backlogs": 0,
}
AVERAGE_STUDENT = {
    "attendance_pct": 75, "study_hours_per_week": 12, "assignment_avg": 65,
    "internal_marks": 30, "prev_exam_score": 60, "assignment_completion": 70,
    "participation_level": 1, "num_backlogs": 1,
}
STRUGGLING_STUDENT = {
    "attendance_pct": 50, "study_hours_per_week": 4, "assignment_avg": 35,
    "internal_marks": 15, "prev_exam_score": 30, "assignment_completion": 40,
    "participation_level": 0, "num_backlogs": 3,
}


# ═════════════════════════════════════════════════════════════════════════════
class TestHealthEndpoint(unittest.TestCase):
    """Tests for GET /health"""

    @classmethod
    def setUpClass(cls):
        cls.client = _app.app.test_client()

    def test_health_returns_200(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)

    def test_health_response_is_json(self):
        r = self.client.get("/health")
        self.assertIsNotNone(r.get_json())

    def test_health_status_field(self):
        r = self.client.get("/health")
        self.assertEqual(r.get_json().get("status"), "healthy")

    def test_health_service_field(self):
        r = self.client.get("/health")
        self.assertEqual(r.get_json().get("service"), "EduPredict AI")

    def test_health_wrong_method_returns_405(self):
        r = self.client.post("/health")
        self.assertEqual(r.status_code, 405)


# ═════════════════════════════════════════════════════════════════════════════
class TestPredictValidInput(unittest.TestCase):
    """Tests for POST /predict with valid student profiles and boundary values."""

    @classmethod
    def setUpClass(cls):
        cls.client = _app.app.test_client()

    # ── Named profiles ────────────────────────────────────────────────────────
    def test_strong_student_returns_200(self):
        r = self.client.post("/predict", json=STRONG_STUDENT)
        self.assertEqual(r.status_code, 200)

    def test_strong_student_category_is_excellent(self):
        r = self.client.post("/predict", json=STRONG_STUDENT)
        self.assertEqual(r.get_json()["performance_category"], "Excellent")

    def test_strong_student_score_above_80(self):
        r = self.client.post("/predict", json=STRONG_STUDENT)
        self.assertGreater(r.get_json()["predicted_score"], 80.0)

    def test_average_student_returns_200(self):
        r = self.client.post("/predict", json=AVERAGE_STUDENT)
        self.assertEqual(r.status_code, 200)

    def test_average_student_category_in_expected_range(self):
        r = self.client.post("/predict", json=AVERAGE_STUDENT)
        cat = r.get_json()["performance_category"]
        self.assertIn(cat, {"Average", "Good"})

    def test_struggling_student_returns_200(self):
        r = self.client.post("/predict", json=STRUGGLING_STUDENT)
        self.assertEqual(r.status_code, 200)

    def test_struggling_student_category_needs_improvement(self):
        r = self.client.post("/predict", json=STRUGGLING_STUDENT)
        self.assertEqual(r.get_json()["performance_category"], "Needs Improvement")

    def test_struggling_student_score_below_50(self):
        r = self.client.post("/predict", json=STRUGGLING_STUDENT)
        self.assertLess(r.get_json()["predicted_score"], 50.0)

    def test_struggling_student_has_8_recommendations(self):
        r = self.client.post("/predict", json=STRUGGLING_STUDENT)
        self.assertEqual(len(r.get_json()["personalized_recommendations"]), 8)

    # ── Boundary values ───────────────────────────────────────────────────────
    def test_all_minimum_values_accepted(self):
        payload = {
            "attendance_pct": 0, "study_hours_per_week": 0, "assignment_avg": 0,
            "internal_marks": 0, "prev_exam_score": 0, "assignment_completion": 0,
            "participation_level": 0, "num_backlogs": 0,
        }
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_all_maximum_values_accepted(self):
        payload = {
            "attendance_pct": 100, "study_hours_per_week": 35, "assignment_avg": 100,
            "internal_marks": 50,  "prev_exam_score": 100, "assignment_completion": 100,
            "participation_level": 2, "num_backlogs": 5,
        }
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_internal_marks_exactly_50_accepted(self):
        payload = dict(STRONG_STUDENT); payload["internal_marks"] = 50
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_num_backlogs_exactly_5_accepted(self):
        payload = dict(STRONG_STUDENT); payload["num_backlogs"] = 5
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_participation_level_0_accepted(self):
        payload = dict(STRONG_STUDENT); payload["participation_level"] = 0
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_participation_level_1_accepted(self):
        payload = dict(STRONG_STUDENT); payload["participation_level"] = 1
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_participation_level_2_accepted(self):
        payload = dict(STRONG_STUDENT); payload["participation_level"] = 2
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_integer_valued_float_participation_accepted(self):
        """2.0 is a whole number; should be accepted as participation_level=2."""
        payload = dict(STRONG_STUDENT); payload["participation_level"] = 2.0
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)

    def test_integer_valued_float_backlogs_accepted(self):
        """0.0 is a whole number; should be accepted as num_backlogs=0."""
        payload = dict(STRONG_STUDENT); payload["num_backlogs"] = 0.0
        r = self.client.post("/predict", json=payload)
        self.assertEqual(r.status_code, 200)


# ═════════════════════════════════════════════════════════════════════════════
class TestPredictValidation(unittest.TestCase):
    """Tests that the API rejects invalid inputs with the correct HTTP codes."""

    @classmethod
    def setUpClass(cls):
        cls.client = _app.app.test_client()

    def _post(self, payload):
        return self.client.post("/predict", json=payload)

    # ── Missing fields ────────────────────────────────────────────────────────
    def test_missing_attendance_pct(self):
        payload = {k: v for k, v in STRONG_STUDENT.items() if k != "attendance_pct"}
        r = self._post(payload)
        self.assertEqual(r.status_code, 422)
        self.assertIn("attendance_pct", r.get_json()["error"])

    def test_missing_study_hours(self):
        payload = {k: v for k, v in STRONG_STUDENT.items() if k != "study_hours_per_week"}
        r = self._post(payload)
        self.assertEqual(r.status_code, 422)

    def test_missing_assignment_avg(self):
        payload = {k: v for k, v in STRONG_STUDENT.items() if k != "assignment_avg"}
        r = self._post(payload)
        self.assertEqual(r.status_code, 422)

    def test_missing_internal_marks(self):
        payload = {k: v for k, v in STRONG_STUDENT.items() if k != "internal_marks"}
        r = self._post(payload)
        self.assertEqual(r.status_code, 422)

    def test_missing_prev_exam_score(self):
        payload = {k: v for k, v in STRONG_STUDENT.items() if k != "prev_exam_score"}
        r = self._post(payload)
        self.assertEqual(r.status_code, 422)

    def test_missing_assignment_completion(self):
        payload = {k: v for k, v in STRONG_STUDENT.items() if k != "assignment_completion"}
        r = self._post(payload)
        self.assertEqual(r.status_code, 422)

    def test_missing_participation_level(self):
        payload = {k: v for k, v in STRONG_STUDENT.items() if k != "participation_level"}
        r = self._post(payload)
        self.assertEqual(r.status_code, 422)

    def test_missing_num_backlogs(self):
        payload = {k: v for k, v in STRONG_STUDENT.items() if k != "num_backlogs"}
        r = self._post(payload)
        self.assertEqual(r.status_code, 422)

    def test_empty_json_object(self):
        r = self.client.post("/predict", json={})
        self.assertEqual(r.status_code, 422)

    # ── Negative values ───────────────────────────────────────────────────────
    def test_negative_attendance(self):
        payload = dict(STRONG_STUDENT); payload["attendance_pct"] = -1
        self.assertEqual(self._post(payload).status_code, 422)

    def test_negative_study_hours(self):
        payload = dict(STRONG_STUDENT); payload["study_hours_per_week"] = -5
        self.assertEqual(self._post(payload).status_code, 422)

    def test_negative_assignment_avg(self):
        payload = dict(STRONG_STUDENT); payload["assignment_avg"] = -10
        self.assertEqual(self._post(payload).status_code, 422)

    def test_negative_internal_marks(self):
        payload = dict(STRONG_STUDENT); payload["internal_marks"] = -1
        self.assertEqual(self._post(payload).status_code, 422)

    def test_negative_prev_exam_score(self):
        payload = dict(STRONG_STUDENT); payload["prev_exam_score"] = -0.1
        self.assertEqual(self._post(payload).status_code, 422)

    def test_negative_backlogs(self):
        payload = dict(STRONG_STUDENT); payload["num_backlogs"] = -1
        self.assertEqual(self._post(payload).status_code, 422)

    def test_negative_participation(self):
        payload = dict(STRONG_STUDENT); payload["participation_level"] = -1
        self.assertEqual(self._post(payload).status_code, 422)

    # ── Above-maximum values ──────────────────────────────────────────────────
    def test_attendance_above_100(self):
        payload = dict(STRONG_STUDENT); payload["attendance_pct"] = 101
        self.assertEqual(self._post(payload).status_code, 422)

    def test_study_hours_above_35(self):
        payload = dict(STRONG_STUDENT); payload["study_hours_per_week"] = 36
        self.assertEqual(self._post(payload).status_code, 422)

    def test_assignment_avg_above_100(self):
        payload = dict(STRONG_STUDENT); payload["assignment_avg"] = 100.1
        self.assertEqual(self._post(payload).status_code, 422)

    def test_internal_marks_above_50(self):
        payload = dict(STRONG_STUDENT); payload["internal_marks"] = 51
        self.assertEqual(self._post(payload).status_code, 422)

    def test_participation_level_3(self):
        payload = dict(STRONG_STUDENT); payload["participation_level"] = 3
        self.assertEqual(self._post(payload).status_code, 422)

    def test_participation_level_10(self):
        payload = dict(STRONG_STUDENT); payload["participation_level"] = 10
        self.assertEqual(self._post(payload).status_code, 422)

    def test_num_backlogs_6(self):
        payload = dict(STRONG_STUDENT); payload["num_backlogs"] = 6
        self.assertEqual(self._post(payload).status_code, 422)

    # ── Non-integer where integer required ────────────────────────────────────
    def test_participation_level_float_15(self):
        payload = dict(STRONG_STUDENT); payload["participation_level"] = 1.5
        self.assertEqual(self._post(payload).status_code, 422)

    def test_num_backlogs_float_27(self):
        payload = dict(STRONG_STUDENT); payload["num_backlogs"] = 2.7
        self.assertEqual(self._post(payload).status_code, 422)

    # ── Non-numeric types ─────────────────────────────────────────────────────
    def test_string_value_rejected(self):
        payload = dict(STRONG_STUDENT); payload["attendance_pct"] = "abc"
        self.assertIn(self._post(payload).status_code, (400, 422))

    def test_null_value_rejected(self):
        payload = dict(STRONG_STUDENT); payload["study_hours_per_week"] = None
        self.assertIn(self._post(payload).status_code, (400, 422))

    def test_boolean_true_rejected(self):
        """JSON true should NOT be accepted as a numeric value (bool subclasses int in Python)."""
        payload = dict(STRONG_STUDENT); payload["assignment_avg"] = True
        self.assertIn(self._post(payload).status_code, (400, 422))

    def test_boolean_false_rejected(self):
        payload = dict(STRONG_STUDENT); payload["internal_marks"] = False
        self.assertIn(self._post(payload).status_code, (400, 422))

    def test_list_value_rejected(self):
        payload = dict(STRONG_STUDENT); payload["prev_exam_score"] = [70]
        self.assertIn(self._post(payload).status_code, (400, 422))

    def test_string_participation_rejected(self):
        payload = dict(STRONG_STUDENT); payload["participation_level"] = "High"
        self.assertIn(self._post(payload).status_code, (400, 422))

    # ── Malformed request body ────────────────────────────────────────────────
    def test_plain_text_body_rejected(self):
        r = self.client.post("/predict", data="hello world", content_type="text/plain")
        self.assertEqual(r.status_code, 400)

    def test_empty_body_rejected(self):
        r = self.client.post("/predict", data="", content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_broken_json_rejected(self):
        r = self.client.post("/predict", data="{bad", content_type="application/json")
        self.assertEqual(r.status_code, 400)


# ═════════════════════════════════════════════════════════════════════════════
class TestResponseStructure(unittest.TestCase):
    """Tests that valid /predict responses have the correct structure and types."""

    @classmethod
    def setUpClass(cls):
        cls.client = _app.app.test_client()
        r = cls.client.post("/predict", json=STRONG_STUDENT)
        cls.strong_data = r.get_json()
        r2 = cls.client.post("/predict", json=STRUGGLING_STUDENT)
        cls.struggling_data = r2.get_json()

    # ── Required keys ─────────────────────────────────────────────────────────
    def test_predicted_score_present(self):
        self.assertIn("predicted_score", self.strong_data)

    def test_performance_category_present(self):
        self.assertIn("performance_category", self.strong_data)

    def test_classification_confidence_present(self):
        self.assertIn("classification_confidence", self.strong_data)

    def test_feature_importance_present(self):
        self.assertIn("feature_importance", self.strong_data)

    def test_personalized_recommendations_present(self):
        self.assertIn("personalized_recommendations", self.strong_data)

    # ── Value types ───────────────────────────────────────────────────────────
    def test_predicted_score_is_float(self):
        self.assertIsInstance(self.strong_data["predicted_score"], float)

    def test_performance_category_is_string(self):
        self.assertIsInstance(self.strong_data["performance_category"], str)

    def test_classification_confidence_is_dict(self):
        self.assertIsInstance(self.strong_data["classification_confidence"], dict)

    def test_feature_importance_is_dict(self):
        self.assertIsInstance(self.strong_data["feature_importance"], dict)

    def test_recommendations_is_list(self):
        self.assertIsInstance(self.strong_data["personalized_recommendations"], list)

    # ── Value ranges and validity ─────────────────────────────────────────────
    def test_predicted_score_in_0_to_100(self):
        score = self.strong_data["predicted_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_performance_category_is_valid(self):
        self.assertIn(self.strong_data["performance_category"], VALID_CATEGORIES)

    def test_confidence_has_4_categories(self):
        self.assertEqual(len(self.strong_data["classification_confidence"]), 4)

    def test_confidence_keys_are_valid_categories(self):
        for key in self.strong_data["classification_confidence"]:
            self.assertIn(key, VALID_CATEGORIES)

    def test_confidence_sums_to_100(self):
        total = sum(self.strong_data["classification_confidence"].values())
        self.assertAlmostEqual(total, 100.0, delta=1.0)

    def test_confidence_values_are_non_negative(self):
        for val in self.strong_data["classification_confidence"].values():
            self.assertGreaterEqual(val, 0.0)

    def test_feature_importance_has_8_entries(self):
        self.assertEqual(len(self.strong_data["feature_importance"]), 8)

    def test_feature_importance_keys_match_feature_cols(self):
        fi_keys = set(self.strong_data["feature_importance"].keys())
        self.assertEqual(fi_keys, set(FEATURE_COLS))

    def test_feature_importance_values_non_negative(self):
        for val in self.strong_data["feature_importance"].values():
            self.assertGreaterEqual(val, 0.0)

    def test_recommendations_non_empty(self):
        self.assertGreater(len(self.strong_data["personalized_recommendations"]), 0)

    def test_recommendations_are_strings(self):
        for rec in self.strong_data["personalized_recommendations"]:
            self.assertIsInstance(rec, str)

    # ── Struggling student specific ───────────────────────────────────────────
    def test_struggling_student_score_in_range(self):
        score = self.struggling_data["predicted_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_struggling_confidence_sums_to_100(self):
        total = sum(self.struggling_data["classification_confidence"].values())
        self.assertAlmostEqual(total, 100.0, delta=1.0)

    def test_struggling_feature_importance_has_8_entries(self):
        self.assertEqual(len(self.struggling_data["feature_importance"]), 8)


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
