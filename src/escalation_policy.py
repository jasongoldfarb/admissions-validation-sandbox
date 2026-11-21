"""
Session 3: Human-in-the-Loop Review Band & Escalation Policy Simulation
Loads Session 2's audited predictions (outputs/audited_applicants.csv),
simulates imperfect human review (human_review_accuracy), compares standard
probability review bands against context-aware escalation, and saves the
trade-off curve for Session 4 visualization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_policy(
    df: pd.DataFrame,
    escalate_mask: pd.Series,
    human_review_accuracy: float = 0.92,
    seed: int = 42,
) -> dict:
    """Apply an escalation mask and simulate human review with realistic accuracy."""
    rng = np.random.default_rng(seed)
    final_pred = df["pred_admit"].to_numpy().copy()

    esc_idx = np.where(escalate_mask.to_numpy())[0]
    true_vals = df["true_qualified"].to_numpy()[esc_idx]

    # Human reviewer gets true readiness right with probability `human_review_accuracy`
    human_correct = rng.binomial(1, human_review_accuracy, size=len(esc_idx))
    human_decisions = np.where(human_correct == 1, true_vals, 1 - true_vals)
    final_pred[esc_idx] = human_decisions

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
        "overall_accuracy": overall_acc,
        "overall_fnr": overall_fnr,
        "trad_accuracy": trad_acc,
        "trad_fnr": trad_fnr,
        "trad_tpr": trad_tpr,
        "nontrad_accuracy": nontrad_acc,
        "nontrad_fnr": nontrad_fnr,
        "nontrad_tpr": nontrad_tpr,
    }


def main() -> None:
    input_path = Path("outputs/audited_applicants.csv")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run `python src/audit_borderline.py` first."
        )

    df = pd.read_csv(input_path)
    human_acc = 0.92

    print("=" * 78)
    print(f"SESSION 3: HUMAN-IN-THE-LOOP ESCALATION SIMULATION (Human Acc = {human_acc:.0%})")
    print("=" * 78)

    # 1. Sweep standard probability bands (for trade-off curve plotting)
    bands = [
        ("No Escalation (Baseline)", 0.50, 0.50),
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
        if low == high:
            mask = pd.Series(False, index=df.index)
        else:
            mask = (df["pred_prob"] >= low) & (df["pred_prob"] <= high)
        stats = evaluate_policy(df, mask, human_review_accuracy=human_acc, seed=42)
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

    # 2. Context-Aware Review Band:
    # Standard uncertainty band [0.20, 0.65] for traditional profiles,
    # lower floor [0.04, 0.65] where quantitative data flattens non-traditional context.
    context_mask = (
        ((df["cohort"] == "traditional") & (df["pred_prob"] >= 0.20) & (df["pred_prob"] <= 0.65))
        | ((df["cohort"] == "non_traditional") & (df["pred_prob"] >= 0.04) & (df["pred_prob"] <= 0.65))
    )
    ctx_stats = evaluate_policy(df, context_mask, human_review_accuracy=human_acc, seed=42)
    ctx_stats["policy"] = "Context-Aware Band (Trad [0.20,0.65] / NonTrad [0.04,0.65])"
    ctx_stats["band_low"] = np.nan
    ctx_stats["band_high"] = np.nan
    curve_rows.append(ctx_stats)

    baseline_nontrad_fnr = curve_rows[0]["nontrad_fnr"]
    fnr_reduction = (baseline_nontrad_fnr - ctx_stats["nontrad_fnr"]) / baseline_nontrad_fnr

    print("\n2. CONTEXT-AWARE ESCALATION POLICY (TARGETED REVIEW BAND)")
    print("-" * 78)
    print(f"Policy: {ctx_stats['policy']}")
    print(f"  Applications Escalated to Human Reader:   {ctx_stats['escalated_pct']:.1%} ({ctx_stats['escalated_count']:,} / {len(df):,})")
    print(f"  Overall System Accuracy:                  {curve_rows[0]['overall_accuracy']:.1%} -> {ctx_stats['overall_accuracy']:.1%}")
    print(f"  Traditional Wrongful Rejection (FNR):     {curve_rows[0]['trad_fnr']:.1%} -> {ctx_stats['trad_fnr']:.1%}")
    print(f"  Non-Traditional Wrongful Rejection (FNR): {baseline_nontrad_fnr:.1%} -> {ctx_stats['nontrad_fnr']:.1%} (-{fnr_reduction:.1%} relative drop)")
    print(f"  Equal Opportunity (TPR Gap):              {curve_rows[0]['trad_tpr']:.1%} vs {curve_rows[0]['nontrad_tpr']:.1%}  -->  {ctx_stats['trad_tpr']:.1%} vs {ctx_stats['nontrad_tpr']:.1%}")
    print("=" * 78)

    tradeoff_df = pd.DataFrame(curve_rows)
    out_path = Path("outputs/escalation_tradeoff.csv")
    tradeoff_df.to_csv(out_path, index=False)
    print(f"\nSaved escalation trade-off results to: {out_path}")


if __name__ == "__main__":
    main()
