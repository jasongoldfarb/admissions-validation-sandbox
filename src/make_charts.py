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

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "seaborn-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)

    bars = ax.bar(labels, values, color=colors, width=0.52, edgecolor="none")
    ax.set_ylim(0, 102)
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "The Illusion of Aggregate Accuracy: Subgroup Blind Spots & Escalation Impact",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#1f2933",
        )

    ax.annotate(
        "High headline accuracy masks\n67.8% missed qualified\nnon-traditional candidates",
        xy=(2, values[2]),
        xytext=(1.55, 81),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdf2e9", edgecolor="#e67e22", lw=1),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_chart2(tradeoff_df: pd.DataFrame, out_path: Path) -> None:
    """Plot human review workload (%) vs. non-traditional wrongful rejection rate."""
    sweep_df = tradeoff_df[~tradeoff_df["policy"].str.startswith("Context-Aware Band")].copy()
    sweep_df = sweep_df.sort_values("escalated_pct")

    ctx_row = tradeoff_df[tradeoff_df["policy"].str.startswith("Context-Aware Band")].iloc[0]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "seaborn-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)

    ax.plot(
        sweep_df["escalated_pct"] * 100,
        sweep_df["nontrad_fnr"] * 100,
        marker="o",
        linewidth=2.2,
        color="#2b5c8f",
        label="Symmetric Probability Band Sweep",
    )

    # Highlight Context-Aware Band point
    ctx_x = ctx_row["escalated_pct"] * 100
    ctx_y = ctx_row["nontrad_fnr"] * 100
    ax.scatter(
        [ctx_x],
        [ctx_y],
        color="#1e8449",
        s=110,
        zorder=5,
        label="Context-Aware Review Band (4.9% Escalated)",
    )
    ax.annotate(
        f"Context-Aware Band\n({ctx_x:.1f}% escalated -> {ctx_y:.1f}% FNR)",
        xy=(ctx_x, ctx_y),
        xytext=(ctx_x + 12, ctx_y + 11),
        arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.3),
        fontsize=9.5,
        fontweight="bold",
        color="#145a32",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eafaf1", edgecolor="#1e8449", lw=1),
    )

    ax.set_xlim(-3, 105)
    ax.set_ylim(0, 76)
    ax.set_xlabel("Applications Escalated to Human Review (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Non-Traditional Wrongful Reject Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Human-in-the-Loop Trade-Off: Escalation Workload vs. Missed Qualified Applicants",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    fig.tight_layout()
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
    make_chart2(tradeoff_df, c2_path)
    print(f"Saved charts to:\n  - {c1_path}\n  - {c2_path}")


if __name__ == "__main__":
    main()
