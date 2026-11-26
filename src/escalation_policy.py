"""
Session 3: Human-in-the-Loop Review Band & Escalation Policy Simulation
Loads Session 2's audited predictions (outputs/audited_applicants.csv),
simulates imperfect human review (human_review_accuracy), compares standard
probability review bands against an observable context-proxy escalation rule,
and saves the trade-off results for Session 4 visualization.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

if __package__:
    from .paths import get_output_dir
    from .selection_policy import DEFAULT_CAPACITY_FRACTION, select_with_capacity
else:
    from paths import get_output_dir
    from selection_policy import DEFAULT_CAPACITY_FRACTION, select_with_capacity


def evaluate_policy(
    df: pd.DataFrame,
    escalate_mask: pd.Series,
    human_review_accuracy: float = 0.92,
    seed: int = 42,
    capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
) -> dict:
    """Apply reviewer overrides, then select eligible applicants within capacity."""
    rng = np.random.default_rng(seed)
    initial_eligibility = df.get("model_eligible", df["pred_admit"]).to_numpy(dtype=bool)
    final_eligibility = initial_eligibility.copy()

    esc_idx = np.where(escalate_mask.to_numpy())[0]
    true_vals = df["true_qualified"].to_numpy()[esc_idx]

    # Human reviewer gets true readiness right with probability `human_review_accuracy`
    human_correct = rng.binomial(1, human_review_accuracy, size=len(esc_idx))
    human_decisions = np.where(human_correct == 1, true_vals, 1 - true_vals)
    final_eligibility[esc_idx] = human_decisions

    final_pred, target_seats = select_with_capacity(
        df["pred_prob"].to_numpy(), final_eligibility, capacity_fraction
    )

    eval_df = df.copy()
    eval_df["final_pred"] = final_pred

    def cohort_stats(sub: pd.DataFrame) -> tuple[float, float, float]:
        acc = accuracy_score(sub["true_qualified"], sub["final_pred"])
        _, _, fn, tp = confusion_matrix(
            sub["true_qualified"], sub["final_pred"], labels=[0, 1]
        ).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return acc, fnr, tpr

    overall_acc, overall_fnr, _ = cohort_stats(eval_df)
    trad_acc, trad_fnr, trad_tpr = cohort_stats(eval_df[eval_df["cohort"] == "traditional"])
    nontrad_acc, nontrad_fnr, nontrad_tpr = cohort_stats(
        eval_df[eval_df["cohort"] == "non_traditional"]
    )

    return {
        "escalated_pct": float(escalate_mask.mean()),
        "escalated_count": int(escalate_mask.sum()),
        "capacity_fraction": capacity_fraction,
        "capacity_seats": target_seats,
        "model_shortlisted_count": int(initial_eligibility.sum()),
        "eligible_count": int(final_eligibility.sum()),
        "admitted_count": int(final_pred.sum()),
        "unfilled_seats": target_seats - int(final_pred.sum()),
        "admitted_pct": float(final_pred.mean()),
        "capacity_fill_pct": float(final_pred.sum() / target_seats) if target_seats else 0.0,
        "overall_accuracy": overall_acc,
        "overall_fnr": overall_fnr,
        "trad_accuracy": trad_acc,
        "trad_fnr": trad_fnr,
        "trad_tpr": trad_tpr,
        "nontrad_accuracy": nontrad_acc,
        "nontrad_fnr": nontrad_fnr,
        "nontrad_tpr": nontrad_tpr,
    }


def main(
    output_dir: Optional[Path] = None,
    capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
) -> None:
    output_dir = get_output_dir(output_dir)
    input_path = output_dir / "audited_applicants.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run `python src/audit_borderline.py` first."
        )

    df = pd.read_csv(input_path)
    human_acc = 0.92

    print("=" * 78)
    print(
        f"SESSION 3: CAPACITY-CONSTRAINED HITL SIMULATION "
        f"(Capacity = {capacity_fraction:.0%}, Human Acc = {human_acc:.0%})"
    )
    print("=" * 78)

    # 1. Sweep standard probability bands
    bands = [
        ("No Escalation (Capacity Baseline)", None, None),
        ("Narrow Band [0.35, 0.65]", 0.35, 0.65),
        ("Moderate Band [0.20, 0.65]", 0.20, 0.65),
        ("Wide Band [0.10, 0.70]", 0.10, 0.70),
        ("Broad Band [0.05, 0.75]", 0.05, 0.75),
        ("Very Broad [0.02, 0.80]", 0.02, 0.80),
        ("Full Human Review [0.00, 1.00]", 0.00, 1.00),
    ]

    curve_rows = []
    print("\n1. STANDARD PROBABILITY REVIEW BAND SWEEP")
    print("-" * 78)
    print(
        f"{'Policy':<31} | {'Escalated':>9} | {'Overall Acc':>11} | "
        f"{'Trad FNR':>8} | {'NonTrad FNR':>11}"
    )
    print("-" * 78)
    for label, low, high in bands:
        if low is None or high is None:
            mask = pd.Series(False, index=df.index)
        else:
            mask = (df["pred_prob"] >= low) & (df["pred_prob"] <= high)
        stats = evaluate_policy(
            df,
            mask,
            human_review_accuracy=human_acc,
            seed=42,
            capacity_fraction=capacity_fraction,
        )
        stats["policy"] = label
        stats["band_low"] = low
        stats["band_high"] = high
        curve_rows.append(stats)

        print(
            f"{label:<31} | {stats['escalated_pct']:>8.1%} | "
            f"{stats['overall_accuracy']:>10.1%} | "
            f"{stats['trad_fnr']:>7.1%} | "
            f"{stats['nontrad_fnr']:>10.1%}"
        )

    # 2. Context-Aware Review Band (Cohort-Blind / Observable Context Proxies):
    # Lower review floor [0.04, 0.65] when observable educational/life constraints
    # exist (school_ap_cap <= 6 OR outside_hours_wk >= 10), else standard [0.20, 0.65].
    has_context_constraint = (df["school_ap_cap"] <= 6) | (df["outside_hours_wk"] >= 10.0)
    context_mask = (
        ((~has_context_constraint) & (df["pred_prob"] >= 0.20) & (df["pred_prob"] <= 0.65))
        | (has_context_constraint & (df["pred_prob"] >= 0.04) & (df["pred_prob"] <= 0.65))
    )
    ctx_stats = evaluate_policy(
        df,
        context_mask,
        human_review_accuracy=human_acc,
        seed=42,
        capacity_fraction=capacity_fraction,
    )
    ctx_stats["policy"] = (
        "Context-Aware Band (AP Cap <=6 or Work >=10h: [0.04,0.65] / Else: [0.20,0.65])"
    )
    ctx_stats["band_low"] = np.nan
    ctx_stats["band_high"] = np.nan
    curve_rows.append(ctx_stats)

    baseline_nontrad_fnr = curve_rows[0]["nontrad_fnr"]
    fnr_reduction = (baseline_nontrad_fnr - ctx_stats["nontrad_fnr"]) / baseline_nontrad_fnr

    print("\n2. CONTEXT-AWARE ESCALATION POLICY (OBSERVABLE CONTEXT PROXIES)")
    print("-" * 78)
    print(f"Policy: {ctx_stats['policy']}")
    print(f"  Applications Escalated to Human Reader:   {ctx_stats['escalated_pct']:.1%} ({ctx_stats['escalated_count']:,} / {len(df):,})")
    print(
        f"  Seats Filled / Target:                    "
        f"{curve_rows[0]['admitted_count']:,}/{curve_rows[0]['capacity_seats']:,} -> "
        f"{ctx_stats['admitted_count']:,}/{ctx_stats['capacity_seats']:,}"
    )
    print(f"  Overall System Accuracy:                  {curve_rows[0]['overall_accuracy']:.1%} -> {ctx_stats['overall_accuracy']:.1%}")
    print(f"  Traditional Wrongful Rejection (FNR):     {curve_rows[0]['trad_fnr']:.1%} -> {ctx_stats['trad_fnr']:.1%}")
    print(f"  Non-Traditional Wrongful Rejection (FNR): {baseline_nontrad_fnr:.1%} -> {ctx_stats['nontrad_fnr']:.1%} (-{fnr_reduction:.1%} relative drop)")
    print(f"  Equal Opportunity (TPR Gap):              {curve_rows[0]['trad_tpr']:.1%} vs {curve_rows[0]['nontrad_tpr']:.1%}  -->  {ctx_stats['trad_tpr']:.1%} vs {ctx_stats['nontrad_tpr']:.1%}")
    print("=" * 78)

    tradeoff_df = pd.DataFrame(curve_rows)
    out_path = output_dir / "escalation_tradeoff.csv"
    tradeoff_df.to_csv(out_path, index=False)
    print(f"\nSaved escalation trade-off results to: {out_path}")


if __name__ == "__main__":
    main()
