"""
generate_dataset.py
-------------------
Generates a synthetic student performance dataset with 500 records.

IMPORTANT: This dataset is entirely SYNTHETIC.
It contains NO real student names, IDs, or personal information.
It is generated programmatically for educational/demonstration purposes only.

Strategy:
    Students are sampled from four realistic profile groups (weak / average /
    good / strong), each with feature ranges that naturally produce scores in
    distinct bands.  The final_score is still computed from a weighted formula
    applied to the raw feature values — the category labels emerge from those
    scores, not from direct injection.  This produces a balanced, realistic
    class distribution while preserving the meaningful relationship between
    features and outcomes required for supervised learning.

Features:
    attendance_pct          : Attendance percentage (0-100)
    study_hours_per_week    : Study hours per week (0-35)
    assignment_avg          : Average assignment score (0-100)
    internal_marks          : Internal assessment marks (0-50)
    prev_exam_score         : Previous exam score (0-100)
    assignment_completion   : Assignment completion percentage (0-100)
    participation_level     : Class participation (encoded: 0=Low, 1=Medium, 2=High)
    num_backlogs            : Number of previous backlogs (0-5)

Target:
    final_score             : Simulated final exam score (0-100, float)
    performance_category    : Derived category label
                              Excellent         (final_score >= 80)
                              Good              (65 <= final_score < 80)
                              Average           (50 <= final_score < 65)
                              Needs Improvement (final_score < 50)
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES   = 500
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "student_data.csv")

# Score formula weights (features normalised to 0-100 scale)
WEIGHTS = {
    "attendance":   0.20,
    "study_hours":  0.22,   # normalised: value/35*100
    "assignment":   0.18,
    "internal":     0.20,   # normalised: value/50*100
    "prev_exam":    0.12,
    "completion":   0.04,
    "participation":0.04,   # normalised: value/2*100
    "backlogs":    -2.5,    # raw count (subtractive)
}

# Four student profile groups — feature ranges chosen so that the weighted
# score formula naturally produces the desired score band for each group.
# All ranges are educationally plausible.
PROFILES = [
    # name,              n,   attend,       study,      assign,      internal,    prev_exam,   completion,   participation, backlogs_probs
    ("Needs Improvement", 90,  (40, 62),  (1,  12),  (35, 58),  (12, 28),  (25, 55),  (30, 65),  [0.60,0.30,0.10],  [0.20,0.30,0.25,0.15,0.07,0.03]),
    ("Average",          165,  (55, 80),  (8,  22),  (48, 75),  (20, 38),  (40, 72),  (45, 82),  [0.30,0.45,0.25],  [0.45,0.30,0.15,0.07,0.02,0.01]),
    ("Good",             155,  (68, 92),  (16, 30),  (62, 90),  (30, 46),  (58, 88),  (60, 95),  [0.15,0.40,0.45],  [0.60,0.25,0.10,0.04,0.01,0.00]),
    ("Excellent",         90,  (82,100),  (25, 35),  (78,100),  (40, 50),  (75,100),  (80,100),  [0.05,0.25,0.70],  [0.80,0.15,0.04,0.01,0.00,0.00]),
]


def _make_group(name, n, attend_r, study_r, assign_r, internal_r,
                prev_r, compl_r, part_p, backlog_p, rng):
    """Generate n student records for one profile group."""
    attendance_pct        = rng.uniform(*attend_r,   n).round(1)
    study_hours           = rng.uniform(*study_r,    n).round(1)
    assignment_avg        = rng.uniform(*assign_r,   n).round(1)
    internal_marks        = rng.uniform(*internal_r, n).round(1)
    prev_exam_score       = rng.uniform(*prev_r,     n).round(1)
    assignment_completion = rng.uniform(*compl_r,    n).round(1)
    participation_level   = rng.choice([0, 1, 2], p=part_p, size=n)
    num_backlogs          = rng.choice([0, 1, 2, 3, 4, 5], p=backlog_p, size=n)

    # Weighted score formula (identical for all groups — no label injection)
    score = (
        WEIGHTS["attendance"]    * attendance_pct
        + WEIGHTS["study_hours"] * (study_hours / 35 * 100)
        + WEIGHTS["assignment"]  * assignment_avg
        + WEIGHTS["internal"]    * (internal_marks / 50 * 100)
        + WEIGHTS["prev_exam"]   * prev_exam_score
        + WEIGHTS["completion"]  * assignment_completion
        + WEIGHTS["participation"] * (participation_level / 2 * 100)
        + WEIGHTS["backlogs"]    * num_backlogs
        + rng.normal(0, 3.0, n)   # realistic noise
    )
    final_score = np.clip(score, 0, 100).round(2)

    return pd.DataFrame({
        "attendance_pct":        attendance_pct,
        "study_hours_per_week":  study_hours,
        "assignment_avg":        assignment_avg,
        "internal_marks":        internal_marks,
        "prev_exam_score":       prev_exam_score,
        "assignment_completion": assignment_completion,
        "participation_level":   participation_level,
        "num_backlogs":          num_backlogs,
        "final_score":           final_score,
    })


def _categorize(s: float) -> str:
    if s >= 80:   return "Excellent"
    if s >= 65:   return "Good"
    if s >= 50:   return "Average"
    return "Needs Improvement"


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    groups = []
    for profile in PROFILES:
        groups.append(_make_group(*profile, rng=rng))

    df = pd.concat(groups, ignore_index=True)

    # Shuffle rows so profile groups are not ordered in the CSV
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Derive category from the computed final_score (not from profile name)
    df["performance_category"] = df["final_score"].apply(_categorize)

    return df


def main():
    print("=" * 55)
    print("  EduPredict AI — Synthetic Dataset Generator")
    print("=" * 55)

    df = generate_dataset()

    print(f"\nDataset shape          : {df.shape}")
    print(f"Features               : {list(df.columns[:-2])}")
    print(f"Targets                : final_score, performance_category")
    print(f"\nClass distribution:")
    counts = df["performance_category"].value_counts()
    for cat, cnt in counts.items():
        print(f"  {cat:<22} : {cnt} ({cnt/len(df)*100:.1f}%)")

    print(f"\nDescriptive statistics:")
    print(df.describe().round(2).to_string())

    print(f"\nMissing values         : {df.isnull().sum().sum()}")
    print(f"Duplicate rows         : {df.duplicated().sum()}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset saved to       : {OUTPUT_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()
