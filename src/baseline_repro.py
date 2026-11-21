"""
Session 1: Baseline Admissions Simulation & Screening Model
Anchored to the empirical priors in Ray, Padmanabhan, and Bouayad
(INFORMS Journal on Computing / arXiv:2304.06901, repo INFORMSJoC/2023.0523):
  - NCES protected/underrepresented share: 1 - 7.8/15.4 (~49.4%)
  - College Board SAT distribution: mean 1030, std 235 (rounded to 10)
  - Equal underlying quality across groups: N(0.5, 0.1)
  - Tier-1 selective admissions cutoff: top 20% (Tau1 = 20)

Translates the paper's group-level evaluation noise gap into concrete
flattened-context features (school AP caps, outside work/caregiving hours,
upward grade trajectory) to test human-in-the-loop escalation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Empirical priors from INFORMSJoC/2023.0523
FRACTION_OF_PROTECTED = 1.0 - (7.8 / 15.4)  # NCES undergraduate enrollment (~49.35%)
SAT_MEAN = 1030.0                            # College Board median/mean
SAT_STD = 235.0                              # College Board std dev
QUALITY_MU = 0.5                             # Equal underlying quality mean (P0 = P1)
QUALITY_SIGMA = 0.1                          # Equal underlying quality std (P0 = P1)
TIER1_ADMIT_RATE = 0.20                      # Tau1 = 20% selective tier cutoff


def generate_applicants(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Simulate applicants using paper priors + flattened qualitative context."""
    rng = np.random.default_rng(seed)

    # Cohort split anchored to NCES share (~49.4% non-traditional / underrepresented)
    is_nontraditional = (rng.uniform(size=n_samples) <= FRACTION_OF_PROTECTED).astype(int)
    cohort = np.where(is_nontraditional == 1, "non_traditional", "traditional")

    # Equal underlying academic & non-academic quality across cohorts: N(0.5, 0.1)
    quality_acad = rng.normal(QUALITY_MU, QUALITY_SIGMA, size=n_samples)
    quality_nonacad = rng.normal(QUALITY_MU, QUALITY_SIGMA, size=n_samples)

    # Standardized latent z-scores for clean feature scaling
    z_acad = (quality_acad - QUALITY_MU) / QUALITY_SIGMA
    z_nonacad = (quality_nonacad - QUALITY_MU) / QUALITY_SIGMA

    # Qualitative context factors flattened in raw tabular snapshots
    school_ap_cap = np.where(
        is_nontraditional == 1,
        rng.integers(2, 7, size=n_samples),
        rng.integers(8, 16, size=n_samples),
    )
    outside_hours_wk = np.where(
        is_nontraditional == 1,
        np.clip(rng.normal(16.0, 6.0, size=n_samples), 0, 35),
        np.clip(rng.normal(3.0, 3.0, size=n_samples), 0, 20),
    )
    upward_trajectory = np.where(
        is_nontraditional == 1,
        rng.binomial(1, 0.40, size=n_samples),
        rng.binomial(1, 0.15, size=n_samples),
    )

    # Observable quantitative features
    gpa = np.clip(
        3.35
        + 0.32 * z_acad
        - 0.008 * outside_hours_wk
        - 0.06 * upward_trajectory
        + rng.normal(0, 0.14, size=n_samples),
        2.0,
        4.0,
    )

    # SAT anchored to College Board N(1030, 235), rounded to nearest 10
    # sqrt(185^2 + 145^2) ~= 235; centered +/-17.5 pt access/prep gap keeps overall mean ~1030
    sat_raw = (
        SAT_MEAN
        + 185.0 * z_acad
        + np.where(is_nontraditional == 1, -17.5, 17.5)
        + rng.normal(0, 145.0, size=n_samples)
    )
    test_score = np.clip(np.round(sat_raw / 10.0) * 10.0, 400, 1600)

    ap_courses_taken = np.clip(
        np.round(
            school_ap_cap * (0.45 + 0.18 * z_acad + rng.normal(0, 0.12, size=n_samples))
        ),
        0,
        school_ap_cap,
    )

    leadership_index = np.clip(
        5.5
        + 1.4 * z_nonacad
        - 0.09 * outside_hours_wk
        + rng.normal(0, 1.2, size=n_samples),
        1.0,
        10.0,
    )

    # Ground-truth readiness (true academic/non-academic quality + context adjustments)
    true_readiness_score = (
        1.05 * z_acad
        + 0.45 * z_nonacad
        + 0.35 * (ap_courses_taken / np.maximum(school_ap_cap, 1))
        + 0.025 * outside_hours_wk
        + 0.35 * upward_trajectory
        + rng.normal(0, 0.35, size=n_samples)
    )
    true_qualified = (
        true_readiness_score >= np.quantile(true_readiness_score, 1.0 - TIER1_ADMIT_RATE)
    ).astype(int)

    # Historical admissions decisions (biased toward raw tabular proxies + familiarity penalty)
    historical_score = (
        2.1 * ((gpa - 3.35) / 0.35)
        + 1.8 * ((test_score - SAT_MEAN) / SAT_STD)
        + 1.1 * ((ap_courses_taken - 4.0) / 3.0)
        + 0.7 * ((leadership_index - 5.5) / 2.0)
        - 0.45 * is_nontraditional
        + rng.normal(0, 0.55, size=n_samples)
    )
    historical_admit = (
        historical_score >= np.quantile(historical_score, 1.0 - TIER1_ADMIT_RATE)
    ).astype(int)

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

    feature_cols = ["gpa", "test_score", "ap_courses_taken", "leadership_index"]

    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["cohort"]
    )
    test_df = test_df.copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, train_df["historical_admit"])

    test_df["pred_prob"] = clf.predict_proba(X_test)[:, 1]
    test_df["pred_admit"] = (test_df["pred_prob"] >= 0.50).astype(int)

    overall = summarize_metrics(test_df)
    trad = summarize_metrics(test_df[test_df["cohort"] == "traditional"])
    nontrad = summarize_metrics(test_df[test_df["cohort"] == "non_traditional"])

    print("=" * 64)
    print("SESSION 1: BASELINE ADMISSIONS SCREENER (PAPER PRIORS)")
    print("=" * 64)
    print(f"Evaluation Holdout Sample: N = {overall['n']:,}")
    print(f"Holdout SAT Mean / SD:     {test_df['test_score'].mean():.1f} / {test_df['test_score'].std():.1f}")
    print(f"Non-Traditional Share:     {test_df['is_nontraditional'].mean():.1%}")
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
