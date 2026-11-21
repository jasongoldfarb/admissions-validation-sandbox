# Selective Admissions Model Validation & Escalation Sandbox

A self-contained Python model-validation sandbox exploring **why strong headline accuracy can hide subgroup blind spots**—and how a **human-in-the-loop review band** ("letting the model say *I'm not sure*") reduces wrongful auto-rejections on applicants whose qualitative context gets flattened by quantitative snapshots.

Built as a low-friction **winter-break / New Year's coding sandbox** for data scientists, analysts, and curious builders.

## Empirical Priors & Simulation Grounding

The applicant generator in `src/baseline_repro.py` is anchored to the empirical priors used in **Ray, Padmanabhan, and Bouayad (INFORMS Journal on Computing, `INFORMSJoC/2023.0523`)**:
- **Subgroup share (NCES):** `1 - 7.8 / 15.4` (~49.4% non-traditional / underrepresented).
- **SAT distribution (College Board):** Mean `1030`, SD `235`, rounded to the nearest 10.
- **Equal underlying talent:** Identical academic and non-academic quality distributions (`N(0.5, 0.1)`) across cohorts.
- **Selectivity threshold:** Top `20%` admit bar (`Tau1 = 20` Tier-1 cutoff).

To test human-in-the-loop escalation at the single-institution level, the sandbox translates the paper's group-level evaluation noise gap into concrete flattened-context variables (school AP caps, outside work/caregiving hours, and upward grade trajectories).

## Key Findings (Simulated Holdout, N = 3,000)

- **Headline accuracy looks reassuring—and even deceptive:** The baseline screener trained on biased historical admissions data reaches **82.5% overall accuracy** (**81.8%** on traditional applicants and **83.2%** on non-traditional applicants).
- **Subgroup blind spot:** Despite higher cohort accuracy, qualified **non-traditional applicants** face a **68.5%** wrongful auto-rejection rate (false negative rate), compared to **12.0%** for traditional applicants.
- **Flattened context:** **206 of the 230** missed non-traditional candidates get pushed into the lowest predicted-probability bucket (`< 0.20`) because raw counts (APs taken, club leadership, cumulative GPA) miss school AP caps, heavy outside work/caregiving hours, and upward grade trajectories.
- **Targeted human escalation:** Pairing a standard `[0.20, 0.65]` review band for traditional applicants with a lower floor `[0.04, 0.65]` for non-traditional applicants (assuming **92%** human review accuracy) escalates just **6.4%** of applications (192 of 3,000), cutting non-traditional wrongful auto-rejections from **68.5% to 49.7%** (a **27.4% relative drop**) and lifting overall accuracy from **82.5% to 85.5%** without changing the underlying scoring model.

## Charts

![Aggregate vs. Subgroup Performance](outputs/chart1_aggregate_vs_subgroup.png)

![Human-in-the-Loop Trade-Off Curve](outputs/chart2_escalation_tradeoff.png)

## Quickstart

```bash
git clone [https://github.com/jasongoldfarb/admissions-validation-sandbox.git](https://github.com/jasongoldfarb/admissions-validation-sandbox.git)
cd admissions-validation-sandbox
pip install -r requirements.txt

python src/baseline_repro.py
python src/audit_borderline.py
python src/escalation_policy.py
python src/make_charts.py
```

## Project Structure

```text
admissions-validation-sandbox/
├── src/
│   ├── baseline_repro.py       # Session 1: Baseline simulation (paper priors) & subgroup metrics
│   ├── audit_borderline.py     # Session 2: Score-zone slice & flattened-context audit
│   ├── escalation_policy.py    # Session 3: Review band + human_review_accuracy simulation
│   └── make_charts.py          # Session 4: Generate PNG visuals from simulation outputs
├── outputs/                    # Generated CSVs and PNG charts
├── requirements.txt
└── README.md
```

## "Try This Next" Winter Break Challenge

1. **Test human review accuracy sensitivity:** Edit `human_acc` in `src/escalation_policy.py` (e.g., `0.85` vs. `0.95`)—at what human review accuracy does escalating borderline cases stop improving overall accuracy?
2. **Hidden Confounder / Omitted Variable Experiment:** Add an unmeasured **legacy / institutional-familiarity boost** into `historical_admit` without exposing it to the model. How does that shift the decision boundary and change which applicants land inside or outside the review band?
3. **Capacity-Constrained Escalation:** Suppose human admissions readers can only review **5%** of total applications. Design a ranking or context-triggered rule to pick *which* 5% get human eyes first.

## Research Attribution

This sandbox is inspired by and anchored to the empirical priors in the systemic fairness research of **Arindam Ray, Balaji Padmanabhan, and Lina Bouayad**:
- *INFORMS Journal on Computing* simulation repository: https://github.com/INFORMSJoC/2023.0523
- *Systemic Fairness* (`arXiv:2304.06901`): https://arxiv.org/pdf/2304.06901.pdf