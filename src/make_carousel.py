from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

OUT_PDF = Path("docs/admissions_validation_carousel.pdf")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

NAVY = "#1B365D"
SLATE = "#4A5568"
GREEN = "#1E7E45"
RED = "#C0392B"
BG = "#FFFFFF"
CARD_BG = "#F7FAFC"
BORDER = "#CBD5E0"

def add_footer(fig, page_num):
    fig.text(0.05, 0.03, "Admissions Validation Sandbox  |  Capacity-Constrained Simulation (20% Seat Cap)",
             fontsize=9, color=SLATE, ha="left")
    fig.text(0.95, 0.03, f"{page_num} / 4", fontsize=9, fontweight="bold", color=NAVY, ha="right")

def make_cover(pdf):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Top accent bar
    ax.add_patch(FancyBboxPatch((0.05, 0.90), 0.90, 0.012, boxstyle="square,pad=0",
                                facecolor=NAVY, edgecolor="none"))

    fig.text(0.05, 0.84, "ADMISSIONS VALIDATION SANDBOX", fontsize=11,
             fontweight="bold", color=GREEN)
    fig.text(0.05, 0.72, "82.5% Accuracy Sounds Great—\nUntil You See Who It’s Missing",
             fontsize=28, fontweight="bold", color=NAVY, linespacing=1.2)
    fig.text(0.05, 0.59,
             "A capacity-constrained Python simulation exploring subgroup blind spots,\n"
             "context-aware human review, and 50-cycle feedback loops in selective admissions.",
             fontsize=13, color=SLATE, linespacing=1.4)

    cards = [
        ("01  THE BLIND SPOT", RED,
         "82.5% headline accuracy masks\na 67.6% wrongful auto-rejection\nrate for qualified non-traditional\napplicants (vs. 11.6% traditional)."),
        ("02  HUMAN REVIEW", GREEN,
         "Escalating just 6.4% of files\n(192 of 3,000) using observable\ncontext lifts accuracy to 84.2%\nand cuts wrongful rejects 5.7 pts."),
        ("03  50-CYCLE DYNAMICS", NAVY,
         "Feedback loops drop wrongful\nrejections over time—but falling\nerror rates alone don’t prove\nthe system is fair.")
    ]

    for i, (title, color, body) in enumerate(cards):
        x = 0.05 + i * 0.31
        ax.add_patch(FancyBboxPatch((x, 0.16), 0.28, 0.34, boxstyle="round,pad=0.015",
                                    facecolor=CARD_BG, edgecolor=BORDER, linewidth=1.2))
        fig.text(x + 0.015, 0.44, title, fontsize=11, fontweight="bold", color=color)
        fig.text(x + 0.015, 0.38, body, fontsize=10.5, color="#2D3748",
                 va="top", linespacing=1.45)

    add_footer(fig, 1)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)

def make_chart_slide(pdf, img_path, kicker, title, takeaway, page_num):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    fig.text(0.05, 0.93, kicker, fontsize=10, fontweight="bold", color=GREEN)
    fig.text(0.05, 0.88, title, fontsize=19, fontweight="bold", color=NAVY)

    img = mpimg.imread(img_path)
    ax_img = fig.add_axes([0.05, 0.19, 0.90, 0.65])
    ax_img.imshow(img)
    ax_img.axis("off")

    ax_box = fig.add_axes([0.05, 0.075, 0.90, 0.095])
    ax_box.axis("off")
    ax_box.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                    facecolor=CARD_BG, edgecolor=BORDER, linewidth=1.0))
    fig.text(0.065, 0.122, takeaway, fontsize=10.5, color="#2D3748", va="center", linespacing=1.35)

    add_footer(fig, page_num)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)

with PdfPages(OUT_PDF) as pdf:
    make_cover(pdf)
    make_chart_slide(
        pdf,
        "outputs/chart1_aggregate_vs_subgroup.png",
        "SLIDE 1 OF 3  |  SINGLE-CYCLE AUDIT",
        "Aggregate Accuracy Hides Subgroup Blind Spots",
        "Takeaway: The baseline screener scores 83.4% accuracy on non-traditional applicants—yet still\n"
        "wrongfully auto-rejects 67.6% of qualified non-traditional candidates under a 20% seat cap.",
        2,
    )
    make_chart_slide(
        pdf,
        "outputs/chart2_escalation_tradeoff.png",
        "SLIDE 2 OF 3  |  CONTEXT-AWARE ESCALATION",
        "Letting the Model Say “I’m Not Sure” Pays Off",
        "Takeaway: Routing 6.4% of files (192 of 3,000) to human review using observable context raises\n"
        "overall accuracy +1.7 pts while cutting wrongful rejections for both groups (-5.7 pts / -2.8 pts).",
        3,
    )
    make_chart_slide(
        pdf,
        "outputs/chart3_longitudinal_dynamics.png",
        "SLIDE 3 OF 3  |  50-CYCLE FEEDBACK SIMULATION",
        "Falling Error Rates Don’t Guarantee Fairness Over Time",
        "Takeaway: Audit who gets missed in the margins, build room for human context, and test how\n"
        "today’s decisions reshape tomorrow’s pool. Full repo & Colab notebook linked in the first comment.",
        4,
    )

print(f"Saved {OUT_PDF}")
