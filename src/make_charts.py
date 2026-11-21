"""
Session 4: Generate Publication-Ready Charts for README & LinkedIn Post
Reads outputs/audited_applicants.csv and outputs/escalation_tradeoff.csv
to produce:
1. outputs/chart1_aggregate_vs_subgroup.png
2. outputs/chart2_escalation_tradeoff.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _set_style() -> None:
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "seaborn-whitegrid"
    )


def make_chart1(audited_df: pd.DataFrame, tradeoff_df: pd.DataFrame, out_path: Path) -> None:
    """Plot headline accuracy vs. subgroup wrongful auto-rejection rates."""
    overall_acc = (audited_df["true_qualified"] == audited_df["pred_admit"]).mean() * 100

    base_row = tradeoff_df[tradeoff_df["policy"] == "No Escalation (Baseline)"].iloc[0]
    ctx_row = tradeoff_df[tradeoff_df["policy"].str.startswith("Context-Aware Band")].iloc[0]

    labels = [
        "Overall Accuracy\n(All Applicants)",
        "Wrongful Rejects\n(Traditional)",
        "Wrongful Rejects\n(Non-Trad Baseline)",
        "Wrongful Rejects\n(Non-Trad + Escalation)",
    ]
    values = [
        overall_acc,
        base_row["trad_fnr"] * 100,
        base_row["nontrad_fnr"] * 100,
        ctx_row["nontrad_fnr"] * 100,
    ]
    colors = ["#2b5c8f", "#5c768d", "#c0392b", "#1e8449"]

    _set_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=200)

    bars = ax.bar(labels, values, color=colors, width=0.52, edgecolor="none")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "The Illusion of Aggregate Accuracy: Subgroup Blind Spots & Escalation Impact",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1f2933",
        )

    ax.annotate(
        f"{overall_acc:.1f}% headline accuracy masks\n{values[2]:.1f}% missed qualified\nnon-traditional candidates",
        xy=(2, values[2] + 6),
        xytext=(1.15, 83),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.3),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf2e9", edgecolor="#e67e22", lw=1),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_chart2(audited_df: pd.DataFrame, tradeoff_df: pd.DataFrame, out_path: Path) -> None:
    """2-panel Cost vs. Payoff storytelling chart for Context-Aware Escalation."""
    base_row = tradeoff_df[tradeoff_df["policy"] == "No Escalation (Baseline)"].iloc[0]
    ctx_row = tradeoff_df[tradeoff_df["policy"].str.startswith("Context-Aware Band")].iloc[0]

    acc_col = next(
        (c for c in ["overall_acc", "overall_accuracy", "accuracy"] if c in tradeoff_df.columns),
        None,
    )
    base_acc = (
        base_row[acc_col] * 100
        if acc_col
        else (audited_df["true_qualified"] == audited_df["pred_admit"]).mean() * 100
    )
    ctx_acc = ctx_row[acc_col] * 100 if acc_col else base_acc

    n_total = len(audited_df)
    esc_pct = ctx_row["escalated_pct"] * 100
    auto_pct = 100.0 - esc_pct
    esc_n = int(round(ctx_row["escalated_pct"] * n_total))

    base_trad_fnr = base_row["trad_fnr"] * 100
    ctx_trad_fnr = ctx_row["trad_fnr"] * 100
    base_nontrad_fnr = base_row["nontrad_fnr"] * 100
    ctx_nontrad_fnr = ctx_row["nontrad_fnr"] * 100
    rel_drop = (base_nontrad_fnr - ctx_nontrad_fnr) / base_nontrad_fnr * 100

    _set_style()
    fig, (ax_cost, ax_payoff) = plt.subplots(
        1, 2, figsize=(11.2, 5.2), dpi=200, gridspec_kw={"width_ratios": [1.0, 2.25]}
    )

    # --- LEFT PANEL: Operational Cost (Human Workload) ---
    ax_cost.bar(["All\nApps"], [auto_pct], width=0.45, color="#d5dce4", label="Automated")
    ax_cost.bar(
        ["All\nApps"],
        [esc_pct],
        bottom=[auto_pct],
        width=0.45,
        color="#1e8449",
        label="Human Review",
    )
    ax_cost.set_ylim(0, 108)
    ax_cost.set_ylabel("Share of Applications (%)", fontsize=10.5, fontweight="bold")
    ax_cost.set_title("1. The Cost: Human Workload", fontsize=11.5, fontweight="bold", pad=10)
    ax_cost.spines["top"].set_visible(False)
    ax_cost.spines["right"].set_visible(False)
    ax_cost.grid(axis="x", visible=False)

    ax_cost.text(
        0,
        auto_pct / 2,
        f"{auto_pct:.1f}%\nAutomated",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#334e68",
    )
    ax_cost.annotate(
        f"Only {esc_pct:.1f}% Escalated\n({esc_n:,} of {n_total:,} apps)",
        xy=(0.22, auto_pct + esc_pct / 2),
        xytext=(0.38, 78),
        arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.4),
        fontsize=9.5,
        fontweight="bold",
        color="#145a32",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eafaf1", edgecolor="#1e8449", lw=1),
    )
    ax_cost.set_xlim(-0.55, 1.35)

    # --- RIGHT PANEL: Validation Payoff (Before vs. After) ---
    metrics = [
        "Overall System\nAccuracy\n(Higher is better)",
        "Traditional\nWrongful Rejects\n(Lower is better)",
        "Non-Traditional\nWrongful Rejects\n(Lower is better)",
    ]
    base_vals = [base_acc, base_trad_fnr, base_nontrad_fnr]
    ctx_vals = [ctx_acc, ctx_trad_fnr, ctx_nontrad_fnr]
    deltas = [
        f"+{ctx_acc - base_acc:.1f} pts",
        f"-{base_trad_fnr - ctx_trad_fnr:.1f} pts",
        f"-{rel_drop:.1f}% relative",
    ]

    y = np.arange(len(metrics))
    h = 0.34

    bars_base = ax_payoff.barh(
        y + h / 2, base_vals, height=h, color="#9fb3c8", label="Baseline (0% Human Review)"
    )
    bars_ctx = ax_payoff.barh(
        y - h / 2,
        ctx_vals,
        height=h,
        color="#1e8449",
        label=f"Context-Aware Escalation ({esc_pct:.1f}% Human)",
    )

    ax_payoff.set_yticks(y)
    ax_payoff.set_yticklabels(metrics, fontsize=10, fontweight="bold")
    ax_payoff.set_xlim(0, 115)
    ax_payoff.set_xlabel("Percentage (%)", fontsize=10.5, fontweight="bold")
    ax_payoff.set_title("2. The Payoff: Accuracy & Fairness", fontsize=11.5, fontweight="bold", pad=10)
    ax_payoff.spines["top"].set_visible(False)
    ax_payoff.spines["right"].set_visible(False)
    ax_payoff.grid(axis="y", visible=False)
    ax_payoff.legend(loc="center right", frameon=True, fontsize=9.5)

    for b_bar, c_bar, b_val, c_val, delta in zip(bars_base, bars_ctx, base_vals, ctx_vals, deltas):
        ax_payoff.text(
            b_val + 1.4,
            b_bar.get_y() + b_bar.get_height() / 2,
            f"{b_val:.1f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#486581",
            fontweight="bold",
        )
        ax_payoff.text(
            c_val + 1.4,
            c_bar.get_y() + c_bar.get_height() / 2,
            f"{c_val:.1f}%  ({delta})",
            va="center",
            ha="left",
            fontsize=9.8,
            color="#145a32",
            fontweight="bold",
        )

    fig.suptitle(
        "Targeted Human-in-the-Loop Escalation: Small Review Lift, Big Fairness Payoff",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    audited_path = Path("outputs/audited_applicants.csv")
    tradeoff_path = Path("outputs/escalation_tradeoff.csv")
    if not audited_path.exists() or not tradeoff_path.exists():
        raise FileNotFoundError(
            "Missing Session 2/3 CSV outputs. Run baseline_repro.py, "
            "audit_borderline.py, and escalation_policy.py first."
        )

    audited_df = pd.read_csv(audited_path)
    tradeoff_df = pd.read_csv(tradeoff_path)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    c1_path = out_dir / "chart1_aggregate_vs_subgroup.png"
    c2_path = out_dir / "chart2_escalation_tradeoff.png"

    make_chart1(audited_df, tradeoff_df, c1_path)
    make_chart2(audited_df, tradeoff_df, c2_path)
    print(f"Saved charts to:\n  - {c1_path}\n  - {c2_path}")


if __name__ == "__main__":
    main()
