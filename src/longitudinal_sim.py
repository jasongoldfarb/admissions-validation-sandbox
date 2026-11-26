"""
Session 5: Longitudinal Multi-Cycle Dynamics Simulation (Table 2 Extension)
Extends the single-cycle admissions audit to t = 50 recurring selection cycles
using the dynamic ecosystem parameters from Ray, Padmanabhan, and Bouayad
(INFORMS Journal on Computing / arXiv:2304.06901, Table 2):
- n = 1,000 individuals
- f = 1 - 7.8/15.4 (~49.4% non-traditional / protected share)
- t = 50 decision cycles
- Tau = 0.80 (top 20% selective approval threshold)
- Delta Q+ = +0.05 (reward for admitted & qualified)
- Delta Q- = -0.05 (penalty for admitted & unqualified)

Compares two parallel regimes starting from the exact same initial population:
1. Automated Baseline (0% HITL)
2. Context-Aware Escalation (cohort-blind observable constraints + 92% HITL)

Writes cycle-by-cycle metrics to outputs/longitudinal_dynamics.csv.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

if __package__:
    from .baseline_repro import (
        FRACTION_OF_PROTECTED,
        QUALITY_MU,
        QUALITY_SIGMA,
        SAT_MEAN,
        SAT_STD,
        TIER1_ADMIT_RATE,
        compute_true_readiness_score,
    )
    from .paths import get_output_dir
    from .selection_policy import DEFAULT_CAPACITY_FRACTION, select_with_capacity
else:
    from baseline_repro import (
        FRACTION_OF_PROTECTED,
        QUALITY_MU,
        QUALITY_SIGMA,
        SAT_MEAN,
        SAT_STD,
        TIER1_ADMIT_RATE,
        compute_true_readiness_score,
    )
    from paths import get_output_dir
    from selection_policy import DEFAULT_CAPACITY_FRACTION, select_with_capacity

FEATURE_COLS = ["gpa", "test_score", "ap_courses_taken", "leadership_index"]
DELTA_Q_POS = 0.05
DELTA_Q_NEG = -0.05


def initialize_ecosystem(n_samples: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """Create the initial population with equal underlying quality across cohorts."""
    rng = np.random.default_rng(seed)

    is_nontraditional = (rng.uniform(size=n_samples) <= FRACTION_OF_PROTECTED).astype(int)
    cohort = np.where(is_nontraditional == 1, "non_traditional", "traditional")

    quality_acad = np.clip(rng.normal(QUALITY_MU, QUALITY_SIGMA, size=n_samples), 0.0, 1.0)
    quality_nonacad = np.clip(rng.normal(QUALITY_MU, QUALITY_SIGMA, size=n_samples), 0.0, 1.0)

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

    return pd.DataFrame(
        {
            "applicant_id": np.arange(1, n_samples + 1),
            "cohort": cohort,
            "is_nontraditional": is_nontraditional,
            "quality_acad": quality_acad,
            "quality_nonacad": quality_nonacad,
            "school_ap_cap": school_ap_cap.astype(int),
            "outside_hours_wk": np.round(outside_hours_wk, 1),
            "upward_trajectory": upward_trajectory.astype(int),
        }
    )


def compute_cycle_features(state_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate observable tabular features and ground-truth readiness for the current cycle."""
    n = len(state_df)
    is_nontrad = state_df["is_nontraditional"].to_numpy()
    school_ap_cap = state_df["school_ap_cap"].to_numpy()
    outside_hours_wk = state_df["outside_hours_wk"].to_numpy()
    upward_trajectory = state_df["upward_trajectory"].to_numpy()

    z_acad = (state_df["quality_acad"].to_numpy() - QUALITY_MU) / QUALITY_SIGMA
    z_nonacad = (state_df["quality_nonacad"].to_numpy() - QUALITY_MU) / QUALITY_SIGMA

    # Observable tabular features (depressed by flattened context for non-traditional applicants)
    gpa = np.clip(
        3.35
        + 0.32 * z_acad
        - 0.008 * outside_hours_wk
        - 0.06 * upward_trajectory
        + rng.normal(0, 0.14, size=n),
        2.0,
        4.0,
    )

    sat_raw = (
        SAT_MEAN
        + 185.0 * z_acad
        + np.where(is_nontrad == 1, -17.5, 17.5)
        + rng.normal(0, 145.0, size=n)
    )
    test_score = np.clip(np.round(sat_raw / 10.0) * 10.0, 400, 1600)

    ap_courses_taken = np.clip(
        np.round(school_ap_cap * (0.45 + 0.18 * z_acad + rng.normal(0, 0.12, size=n))),
        0,
        school_ap_cap,
    )

    leadership_index = np.clip(
        5.5 + 1.4 * z_nonacad - 0.09 * outside_hours_wk + rng.normal(0, 1.2, size=n),
        1.0,
        10.0,
    )

    true_readiness_score = compute_true_readiness_score(
        z_acad,
        z_nonacad,
        ap_courses_taken / np.maximum(school_ap_cap, 1),
        outside_hours_wk,
        upward_trajectory,
        rng,
    )
    true_qualified = (
        true_readiness_score >= np.quantile(true_readiness_score, 1.0 - TIER1_ADMIT_RATE)
    ).astype(int)

    historical_score = (
        2.1 * ((gpa - 3.35) / 0.35)
        + 1.8 * ((test_score - SAT_MEAN) / SAT_STD)
        + 1.1 * ((ap_courses_taken - 4.0) / 3.0)
        + 0.7 * ((leadership_index - 5.5) / 2.0)
        - 0.45 * is_nontrad
        + rng.normal(0, 0.55, size=n)
    )
    historical_admit = (
        historical_score >= np.quantile(historical_score, 1.0 - TIER1_ADMIT_RATE)
    ).astype(int)

    out = state_df.copy()
    out["gpa"] = np.round(gpa, 3)
    out["test_score"] = test_score.astype(int)
    out["ap_courses_taken"] = ap_courses_taken.astype(int)
    out["leadership_index"] = np.round(leadership_index, 2)
    out["true_qualified"] = true_qualified
    out["historical_admit"] = historical_admit
    return out


def apply_quality_updates(
    quality: np.ndarray,
    true_qualified: np.ndarray,
    final_admit: np.ndarray,
    delta_q_pos: float = DELTA_Q_POS,
    delta_q_neg: float = DELTA_Q_NEG,
    decay: float = 0.08,
    shock_std: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Update underlying quality: +delta_q_pos (TP), delta_q_neg (FP), plus mild mean-reversion."""
    updated = quality.astype(float).copy()
    tp_mask = (final_admit == 1) & (true_qualified == 1)
    fp_mask = (final_admit == 1) & (true_qualified == 0)

    updated[tp_mask] += delta_q_pos
    updated[fp_mask] += delta_q_neg

    # Mild mean-reversion toward QUALITY_MU prevents artificial ceiling lock-in over 50 cycles
    if decay > 0.0:
        updated = updated - decay * (updated - QUALITY_MU)
    if shock_std > 0.0 and rng is not None:
        updated = updated + rng.normal(0.0, shock_std, size=len(updated))

    return np.clip(updated, 0.0, 1.0)


def _cohort_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0


def run_simulation(
    n_samples: int = 1_000,
    n_cycles: int = 50,
    regime: str = "baseline",
    human_review_accuracy: float = 0.92,
    delta_q_pos: float = DELTA_Q_POS,
    delta_q_neg: float = DELTA_Q_NEG,
    decay: float = 0.08,
    shock_std: float = 0.02,
    retrain: bool = True,
    seed: int = 42,
    capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
) -> pd.DataFrame:
    """Run a capacity-constrained simulation under baseline or context-aware review."""
    if regime not in {"baseline", "context_aware_hitl"}:
        raise ValueError(f"Unknown regime '{regime}'. Use 'baseline' or 'context_aware_hitl'.")

    state_df = initialize_ecosystem(n_samples=n_samples, seed=seed)
    feature_rng = np.random.default_rng(seed + 100)
    review_rng = np.random.default_rng(seed + 200)
    shock_rng = np.random.default_rng(seed + 300)

    # Fit initial screener on Cycle 0 historical-biased labels
    init_df = compute_cycle_features(state_df, feature_rng)
    train_X = init_df[FEATURE_COLS].to_numpy()
    train_y = init_df["historical_admit"].to_numpy()

    scaler = StandardScaler()
    clf = LogisticRegression(random_state=seed)
    clf.fit(scaler.fit_transform(train_X), train_y)

    records = []
    for cycle in range(1, n_cycles + 1):
        cycle_df = compute_cycle_features(state_df, feature_rng)
        X_cur = cycle_df[FEATURE_COLS].to_numpy()
        pred_prob = clf.predict_proba(scaler.transform(X_cur))[:, 1]
        model_eligibility, target_seats = select_with_capacity(
            pred_prob, np.ones(n_samples, dtype=bool), capacity_fraction
        )
        final_eligibility = model_eligibility.copy()

        if regime == "context_aware_hitl":
            has_constraint = (cycle_df["school_ap_cap"] <= 6) | (
                cycle_df["outside_hours_wk"] >= 10.0
            )
            escalate_mask = (
                ((~has_constraint) & (pred_prob >= 0.20) & (pred_prob <= 0.65))
                | (has_constraint & (pred_prob >= 0.04) & (pred_prob <= 0.65))
            ).to_numpy()
        else:
            escalate_mask = np.zeros(n_samples, dtype=bool)

        esc_idx = np.where(escalate_mask)[0]
        if len(esc_idx) > 0:
            true_vals = cycle_df["true_qualified"].to_numpy()[esc_idx]
            human_correct = review_rng.binomial(1, human_review_accuracy, size=len(esc_idx))
            final_eligibility[esc_idx] = np.where(human_correct == 1, true_vals, 1 - true_vals)

        final_admit, _ = select_with_capacity(
            pred_prob, final_eligibility, capacity_fraction
        )

        y_true = cycle_df["true_qualified"].to_numpy()
        trad_mask = (cycle_df["cohort"] == "traditional").to_numpy()
        nontrad_mask = ~trad_mask

        trad_q = float(state_df.loc[trad_mask, "quality_acad"].mean())
        nontrad_q = float(state_df.loc[nontrad_mask, "quality_acad"].mean())

        records.append(
            {
                "regime": regime,
                "cycle": cycle,
                "trad_mean_q": trad_q,
                "nontrad_mean_q": nontrad_q,
                "quality_gap": trad_q - nontrad_q,
                "overall_accuracy": float(accuracy_score(y_true, final_admit)),
                "trad_fnr": _cohort_fnr(y_true[trad_mask], final_admit[trad_mask]),
                "nontrad_fnr": _cohort_fnr(y_true[nontrad_mask], final_admit[nontrad_mask]),
                "escalated_pct": float(escalate_mask.mean()),
                "capacity_fraction": capacity_fraction,
                "capacity_seats": target_seats,
                "eligible_count": int(final_eligibility.sum()),
                "admitted_count": int(final_admit.sum()),
                "unfilled_seats": target_seats - int(final_admit.sum()),
                "admitted_pct": float(final_admit.mean()),
                "capacity_fill_pct": (
                    float(final_admit.sum() / target_seats) if target_seats else 0.0
                ),
            }
        )

        # Compound quality for next cycle
        state_df["quality_acad"] = apply_quality_updates(
            state_df["quality_acad"].to_numpy(),
            y_true,
            final_admit,
            delta_q_pos=delta_q_pos,
            delta_q_neg=delta_q_neg,
            decay=decay,
            shock_std=shock_std,
            rng=shock_rng,
        )

        # Retrain on cumulative observed outcomes of admitted candidates
        if retrain:
            admitted_mask = final_admit == 1
            if admitted_mask.any():
                train_X = np.vstack([train_X, X_cur[admitted_mask]])
                train_y = np.concatenate([train_y, y_true[admitted_mask]])
                if len(np.unique(train_y)) == 2:
                    scaler = StandardScaler()
                    clf = LogisticRegression(random_state=seed)
                    clf.fit(scaler.fit_transform(train_X), train_y)

    return pd.DataFrame(records)


def main(
    output_dir: Optional[Path] = None,
    capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
) -> None:
    base_df = run_simulation(regime="baseline", seed=42, capacity_fraction=capacity_fraction)
    hitl_df = run_simulation(
        regime="context_aware_hitl", seed=42, capacity_fraction=capacity_fraction
    )
    results = pd.concat([base_df, hitl_df], ignore_index=True)

    print("=" * 76)
    print("SESSION 5: LONGITUDINAL DYNAMICS (n=1,000, t=50, dQ+=+0.05, dQ-=-0.05)")
    print("=" * 76)
    for label, sub in [("Automated Baseline (0% HITL)", base_df), ("Context-Aware Escalation", hitl_df)]:
        c1, c50 = sub.iloc[0], sub.iloc[-1]
        print(f"\n{label}:")
        print(f"  Quality Gap (Trad - NonTrad): {c1['quality_gap']:+.4f} (t=1) -> {c50['quality_gap']:+.4f} (t=50)")
        print(f"  Non-Trad Wrongful Reject FNR: {c1['nontrad_fnr']:>6.1%} (t=1) -> {c50['nontrad_fnr']:>6.1%} (t=50)")
        print(f"  Overall System Accuracy:      {c1['overall_accuracy']:>6.1%} (t=1) -> {c50['overall_accuracy']:>6.1%} (t=50)")
        print(
            f"  Seats Filled / Target:        "
            f"{c1['admitted_count']:,}/{c1['capacity_seats']:,} (t=1) -> "
            f"{c50['admitted_count']:,}/{c50['capacity_seats']:,} (t=50)"
        )
        print(f"  Mean Escalation Workload:     {sub['escalated_pct'].mean():>6.1%}")
    print("=" * 76)

    output_dir = get_output_dir(output_dir)
    out_path = output_dir / "longitudinal_dynamics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nSaved longitudinal dynamics to: {out_path}")


if __name__ == "__main__":
    main()
