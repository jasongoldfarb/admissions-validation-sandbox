"""
Session 2: Borderline & Subgroup Context Audit
Loads the holdout predictions from Session 1 (outputs/baseline_applicants.csv),
slices error rates across probability zones, and quantifies how flattened
qualitative context drives wrongful auto-rejections on borderline applicants.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

if __package__:
    from .paths import get_output_dir
else:
    from paths import get_output_dir


def assign_zone(prob: float, low: float = 0.20, high: float = 0.65) -> str:
    if prob < low:
        return "1_clear_reject (<0.20)"
    if prob <= high:
        return "2_borderline (0.20-0.65)"
    return "3_clear_admit (>0.65)"


def main(output_dir: Optional[Path] = None) -> None:
    output_dir = get_output_dir(output_dir)
    input_path = output_dir / "baseline_applicants.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run `python src/baseline_repro.py` first."
        )

    df = pd.read_csv(input_path)
    df["zone"] = df["pred_prob"].apply(assign_zone)
    df["ap_utilization"] = df["ap_courses_taken"] / df["school_ap_cap"].clip(lower=1)
    df["wrongful_reject"] = ((df["true_qualified"] == 1) & (df["pred_admit"] == 0)).astype(int)

    print("=" * 72)
    print("SESSION 2: BORDERLINE BAND & QUALITATIVE CONTEXT AUDIT")
    print("=" * 72)

    # 1. Zone breakdown across the whole pool
    print("\n1. SCORE DISTRIBUTION & ERROR CONCENTRATION BY DECISION ZONE")
    print("-" * 72)
    zone_summary = (
        df.groupby("zone", as_index=False)
        .agg(
            applicants=("applicant_id", "count"),
            share_of_pool=("applicant_id", lambda s: len(s) / len(df)),
            true_qualified=("true_qualified", "sum"),
            auto_rejected=("pred_admit", lambda s: int((s == 0).sum())),
            wrongful_rejects=("wrongful_reject", "sum"),
        )
    )
    for _, row in zone_summary.iterrows():
        print(
            f"{row['zone']:<26} | N={int(row['applicants']):>4} ({row['share_of_pool']:>5.1%}) "
            f"| True Qual={int(row['true_qualified']):>3} "
            f"| Wrongful Rejects={int(row['wrongful_rejects']):>3}"
        )

    # 2. Cohort x Zone audit for qualified applicants
    print("\n2. WHERE QUALIFIED APPLICANTS GET MISSED (COHORT x ZONE)")
    print("-" * 72)
    qual_df = df[df["true_qualified"] == 1].copy()
    for cohort in ["traditional", "non_traditional"]:
        sub = qual_df[qual_df["cohort"] == cohort]
        total_qual = len(sub)
        total_missed = int(sub["wrongful_reject"].sum())
        print(f"\nCohort: {cohort.upper()} (Total Qualified = {total_qual}, Total Missed = {total_missed} [{total_missed/total_qual:.1%}])")
        for zone, z_df in sub.groupby("zone"):
            missed = int(z_df["wrongful_reject"].sum())
            print(
                f"  {zone:<26} | Qualified in Zone={len(z_df):>3} "
                f"| Wrongfully Auto-Rejected={missed:>3} ({missed/total_qual:>5.1%} of cohort qualified)"
            )

    # 3. Qualitative context comparison: Wrongfully rejected non-traditional vs. auto-admitted traditional
    missed_nontrad = df[(df["cohort"] == "non_traditional") & (df["wrongful_reject"] == 1)]
    admitted_trad = df[(df["cohort"] == "traditional") & (df["true_qualified"] == 1) & (df["pred_admit"] == 1)]

    print("\n3. FLATTENED CONTEXT SNAPSHOT (WHAT THE TABULAR MODEL MISSED)")
    print("-" * 72)
    metrics = [
        ("Mean GPA (Observable)", "gpa", "{:.2f}"),
        ("Mean Test Score (Observable)", "test_score", "{:.0f}"),
        ("Raw AP Courses Taken (Observable)", "ap_courses_taken", "{:.1f}"),
        ("School AP Offerings Cap (Hidden Context)", "school_ap_cap", "{:.1f}"),
        ("AP Utilization Rate (Hidden Context)", "ap_utilization", "{:.1%}"),
        ("Outside Work/Care Hours/Wk (Hidden Context)", "outside_hours_wk", "{:.1f} hrs"),
        ("Upward Grade Trajectory Share (Hidden Context)", "upward_trajectory", "{:.1%}"),
    ]

    print(f"{'Feature / Context Signal':<46} | {'Missed Non-Trad':>15} | {'Admitted Trad':>14}")
    print("-" * 72)
    for label, col, fmt in metrics:
        v1 = fmt.format(missed_nontrad[col].mean())
        v2 = fmt.format(admitted_trad[col].mean())
        print(f"{label:<46} | {v1:>15} | {v2:>14}")

    print("=" * 72)
    output_path = output_dir / "audited_applicants.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved audited dataset with decision zones to: {output_path}")


if __name__ == "__main__":
    main()
