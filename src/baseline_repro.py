"""
Session 1: Baseline Admissions Simulation & Screening Model
Inspired by the multi-characteristic simulation design in Ray, Padmanabhan,
and Bouayad (INFORMS Journal on Computing / arXiv:2304.06901).

Generates synthetic applicant data where quantitative snapshots flatten
qualitative context for non-traditional applicants, trains a baseline
automated screener on historical admissions outcomes, and logs baseline
accuracy vs. subgroup error rates.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_applicants(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Simulate selective university applicants with observable & latent context."""
    rng = np.random.default_rng(seed)

    # Cohort split: 82% traditional feeder/high-resource profiles, 18% non-traditional
    is_nontraditional = rng.binomial(1, 0.18, size=n_samples)
    cohort = np.where(is_nontraditional == 1, "non_traditional", "traditional")

    # Latent academic potential (equal underlying capability distribution across cohorts)
    latent_ability = rng.normal(0.0, 1.0, size=n_samples)

    # Qualitative context factors (often flattened or absent in tabular snapshots)
    # 1) School AP availability (fewer APs offered at under-resourced schools)
    school_ap_cap = np.where(
        is_nontraditional == 1,
        rng.integers(2, 7, size=n_samples),
        rng.integers(8, 16, size=n_samples),
    )
    # 2) Outside responsibilities (part-time jobs, family caregiving, non-varsity commitments)
    outside_hours_wk = np.where(
        is_nontraditional == 1,
        np.clip(rng.normal(16.0, 6.0, size=n_samples), 0, 35),
        np.clip(rng.normal(3.0, 3.0, size=n_samples), 0, 20),
    )
    # 3) Upward grade trajectory (+1 indicates strong junior/senior rebound)
    upward_trajectory = np.where(
        is_nontraditional == 1,
        rng.binomial(1, 0.40, size=n_samples),
        rng.binomial(1, 0.15, size=n_samples),
    )

    # Observable quantitative features (affected by environment & outside load)
    gpa = np.clip(
        3.55
        + 0.28 * latent_ability
        - 0.008 * outside_hours_wk
        - 0.06 * upward_trajectory  # freshman dip depresses cumulative GPA
        + rng.normal(0, 0.14, size=n_samples),
        2.0,
        4.0,
    )

    test_score = np.clip(
        1320
        + 115 * latent_ability
        - 35 * is_nontraditional  # test prep / access gap
        + rng.normal(0, 55, size=n_samples),
        800,
        1600,
    ).round()

    ap_courses_taken = np.clip(
        np.round(
            school_ap_cap * (0.45 + 0.18 * latent_ability + rng.normal(0, 0.12, size=n_samples))
        ),
        0,
        school_ap_cap,
    )

    leadership_index = np.clip(
        5.5
        + 1.4 * latent_ability
        - 0.09 * outside_hours_wk  # paid work/caregiving crowds out traditional clubs
        + rng.normal(0, 1.2, size=n_samples),
        1.0,
        10.0,
    )

    # Ground-truth readiness (incorporates true ability + grit/context adjustments)
    true_readiness_score = (
        1.35 * latent_ability
        + 0.35 * (ap_courses_taken / np.maximum(school_ap_cap, 1))
        + 0.025 * outside_hours_wk
        + 0.35 * upward_trajectory
        + rng.normal(0, 0.35, size=n_samples)
    )
    # ~28% selective admit bar
    true_qualified = (true_readiness_score >= np.quantile(true_readiness_score, 0.72)).astype(int)

    # Historical admissions decisions (biased toward raw quantitative proxies)
    historical_score = (
        2.1 * ((gpa - 3.5) / 0.3)
        + 1.8 * ((test_score - 1300) / 120)
        + 1.1 * ((ap_courses_taken - 4) / 3)
        + 0.7 * ((leadership_index - 5.5) / 2)
        - 0.45 * is_nontraditional  # historical familiarity / institutional bias
        + rng.normal(0, 0.55, size=n_samples)
    )
    historical_admit = (historical_score >= np.quantile(historical_score, 0.72)).astype(int)

    return pd.DataFrame(
        {
            "applicant_id": np.arange(1, n_samples + 1),
            "cohort": cohort,
            "is_nontraditional": is_nontraditional,
            "gpa": np.round(gpa, 3),
            "test_score": test_score.astype(int),
            "ap_courses_taken": ap_courses_taken.astype(int),
            "school_ap_cap": school_ap_cap.astype(int),
            "leadership_index": np.round(leadership_index, 2),
            "outside_hours_wk": np.round(outside_hours_wk, 1),
            "upward_trajectory": upward_trajectory,
            "true_qualified": true_qualified,
            "historical_admit": historical_admit,
        }
    )


def summarize_metrics(df: pd.DataFrame, label_col: str = "true_qualified", pred_col: str = "pred_admit") -> dict:
    """Compute accuracy, false negative rate (missed qualified), and true positive rate."""
    acc = accuracy_score(df[label_col], df[pred_col])
    tn, fp, fn, tp = confusion_matrix(df[label_col], df[pred_col], labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "n": len(df),
        "accuracy": acc,
        "fnr_wrongful_reject": fnr,
        "tpr_equal_opportunity": tpr,
        "missed_qualified_count": int(fn),
    }


def main() -> None:
    df = generate_applicants(n_samples=10_000, seed=42)

    # Observable quantitative features available to the automated screener
    feature_cols = ["gpa", "test_score", "ap_courses_taken", "leadership_index"]

    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["cohort"]
    )
    test_df = test_df.copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    # Baseline screener trained on historical admissions outcomes
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, train_df["historical_admit"])

    test_df["pred_prob"] = clf.predict_proba(X_test)[:, 1]
    test_df["pred_admit"] = (test_df["pred_prob"] >= 0.50).astype(int)

    # Evaluate against ground-truth readiness
    overall = summarize_metrics(test_df)
    trad = summarize_metrics(test_df[test_df["cohort"] == "traditional"])
    nontrad = summarize_metrics(test_df[test_df["cohort"] == "non_traditional"])

    print("=" * 64)
    print("SESSION 1: BASELINE ADMISSIONS SCREENER REPRODUCTION")
    print("=" * 64)
    print(f"Evaluation Holdout Sample: N = {overall['n']:,}")
    print(f"Overall Accuracy (vs. True Readiness):      {overall['accuracy']:.1%}")
    print(f"Overall Wrongful Auto-Rejection Rate (FNR): {overall['fnr_wrongful_reject']:.1%}")
    print("-" * 64)
    print(f"Traditional Cohort (n = {trad['n']:,}):")
    print(f"  Accuracy:                                 {trad['accuracy']:.1%}")
    print(f"  Wrongful Auto-Rejection Rate (FNR):       {trad['fnr_wrongful_reject']:.1%}")
    print(f"  True Positive Rate (Equal Opportunity):   {trad['tpr_equal_opportunity']:.1%}")
    print("-" * 64)
    print(f"Non-Traditional Cohort (n = {nontrad['n']:,}):")
    print(f"  Accuracy:                                 {nontrad['accuracy']:.1%}")
    print(f"  Wrongful Auto-Rejection Rate (FNR):       {nontrad['fnr_wrongful_reject']:.1%}")
    print(f"  True Positive Rate (Equal Opportunity):   {nontrad['tpr_equal_opportunity']:.1%}")
    print("=" * 64)

    output_path = Path("outputs/baseline_applicants.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    print(f"\nSaved holdout dataset & baseline scores to: {output_path}")


if __name__ == "__main__":
    main()
