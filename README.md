# Selective Admissions Model Validation & Escalation Sandbox

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jasongoldfarb/admissions-validation-sandbox/blob/main/walkthrough.ipynb)

A self-contained Python model-validation sandbox exploring **why strong headline accuracy can hide subgroup blind spots**—and how a **human-in-the-loop review band** ("letting the model say *I'm not sure*") reduces wrongful auto-rejections on applicants whose qualitative context gets flattened by quantitative snapshots.

👉 **Want to explore interactively?** Open [`walkthrough.ipynb`](walkthrough.ipynb) directly on GitHub to read the pre-rendered tables and charts, or click the **Open in Colab** badge above to run the operational sensitivity sandbox in your browser with zero local setup.

Built as a low-friction **winter-break / New Year's coding sandbox** for data scientists, analysts, and curious builders. It uses synthetic data for training and is not an admissions decision system.

## Empirical Priors & Simulation Grounding

The applicant generator in `src/baseline_repro.py` and the 50-cycle extension in `src/longitudinal_sim.py` are anchored to the empirical priors and Table 2 parameters used in **Ray, Padmanabhan, and Bouayad (INFORMS Journal on Computing, `INFORMSJoC/2023.0523`)**:
- **Subgroup share (NCES):** `1 - 7.8 / 15.4` (~49.4% non-traditional / underrepresented).
- **SAT distribution (College Board):** Mean `1030`, SD `235`, rounded to the nearest 10.
- **Equal underlying talent:** Identical academic and non-academic quality distributions (`N(0.5, 0.1)`) across cohorts.
- **Readiness benchmark:** Top `20%` labeled qualified (`Tau1 = 20` Tier-1 cutoff); this ground-truth label is distinct from the configurable seat capacity.
- **Longitudinal dynamics (Table 2):** `n = 1,000` individuals, `t = 50` recurring selection cycles, `∆Q+ = +0.05` (reward for admitted & qualified), `∆Q- = -0.05` (penalty for admitted & unqualified).

To test human-in-the-loop escalation at the single-institution level, the sandbox translates the paper's group-level evaluation noise gap into concrete flattened-context variables (school AP caps, outside work/caregiving hours, and upward grade trajectories). Both the single-cycle and longitudinal examples use the same readiness definition: latent academic and non-academic quality, AP utilization, outside work/care hours, upward trajectory, and a noise term, with the top 20% labeled qualified each cycle.

## Key Findings (Simulated Holdout, N = 3,000; Capacity = 20%)

- **Aggregate metrics conceal subgroup errors:** Overall accuracy is **82.5%** (**81.7%** traditional; **83.4%** non-traditional). Yet the non-traditional FNR is **67.6%**, versus **11.6%** traditional.
- **Flattened context:** **206 of 227** missed qualified non-traditional candidates fall in the lowest probability zone (`< 0.20`), where school AP availability, work/care responsibilities, and grade trajectory are not captured by raw feature counts.
- **Capacity is separate from the model score:** The model ranks applicants; by default the simulation targets **600 admits from 3,000**. The baseline selects the highest-scoring applicants up to that cap. The default contextual policy reviews **192 applicants (6.4%)**, fills all **600 seats**, and changes overall accuracy from **82.5% to 84.2%**. Non-traditional FNR moves from **67.6% to 61.9%** and traditional FNR from **11.6% to 8.8%**. The reviewer may change eligibility, but cannot increase the class beyond the configured cap.
- **Capacity-constrained longitudinal dynamics (`src/longitudinal_sim.py`):** With 200 seats per 1,000 people each cycle, both regimes fill the capacity in the default seeded run. Baseline non-traditional FNR moves from `63.3%` (cycle 1) to `3.5%` (cycle 50), and its quality gap moves from `+0.0134` to `-0.0302`. Context-aware review averages `18.3%`; its non-traditional FNR moves from `55.8%` to `0.8%`, with quality gap moving from `+0.0134` to `-0.0371`. These synthetic outcomes are illustrative, not evidence that the policy is fair in practice.

## Charts

![Aggregate vs. Subgroup Performance](outputs/chart1_aggregate_vs_subgroup.png)

![Targeted Human-in-the-Loop Escalation: Cost vs. Payoff](outputs/chart2_escalation_tradeoff.png)

![50-Cycle Longitudinal Dynamics](outputs/chart3_longitudinal_dynamics.png)

## Quickstart

Requires Python 3.9 or newer. The dependency ranges include the notebook and test tools.

```bash
git clone https://github.com/jasongoldfarb/admissions-validation-sandbox.git
cd admissions-validation-sandbox
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the complete workflow in dependency order:

```bash
python src/run_pipeline.py
python -m pytest -v
```

The default capacity is `20%` of the evaluation pool. Change it with `--capacity-fraction` (a value from `0` to `1`); probabilities rank applicants and the capacity limits the final selected count:

```bash
python src/run_pipeline.py --capacity-fraction 0.10 --output-dir /tmp/admissions-sandbox-run
```

The model's historical `0.50` binary cutoff is no longer the admission decision. The baseline shortlist is the top-scoring capacity-sized group; for HITL, reviewer decisions can change shortlist eligibility before the final score ranking and capacity limit are applied.

The selected count is the nearest whole applicant to `pool size × capacity fraction`; score ties are resolved by input order. If review leaves too few eligible applicants, seats can remain unfilled. Individual scripts also expose a `capacity_fraction` parameter; when invoked by their script path, they read and write the repository's `outputs/` directory regardless of the current working directory. To explore interactively, launch the notebook locally:

```bash
jupyter notebook walkthrough.ipynb
```

## Project Structure

For module-level documentation, function signatures, and parameter details, see [`src/README.md`](src/README.md).

```text
admissions-validation-sandbox/
├── src/
│   ├── __init__.py             # Enables clean module imports from walkthrough.ipynb
│   ├── README.md               # Module-level reference for pipeline modules and runner
│   ├── baseline_repro.py       # Session 1: Baseline simulation (paper priors) & subgroup metrics
│   ├── audit_borderline.py     # Session 2: Score-zone slice & flattened-context audit
│   ├── escalation_policy.py    # Session 3: Observable context-proxy review band simulation
│   ├── make_charts.py          # Session 4: Generate PNG visuals from simulation outputs
│   ├── longitudinal_sim.py     # Session 5: 50-cycle longitudinal dynamics (Table 2 extension)
│   ├── paths.py                # Shared repository-relative output path handling
│   ├── selection_policy.py     # Shared capacity-limited applicant ranking
│   └── run_pipeline.py         # One-command end-to-end runner
├── tests/
│   ├── test_baseline_repro.py
│   ├── test_audit_borderline.py
│   ├── test_escalation_policy.py
│   ├── test_make_charts.py
│   ├── test_longitudinal_sim.py
│   ├── test_selection_policy.py
│   └── test_run_pipeline.py
├── outputs/                    # Generated CSVs and PNG charts
├── walkthrough.ipynb           # End-to-end narrated walkthrough & operational sensitivity sandbox
├── requirements.txt
└── README.md
```

## Running Tests

Run the `pytest` suite (tests cover seed reproducibility, schema/bounds, longitudinal assumptions and updates, edge cases, chart generation, and the full pipeline from outside the repository; generated test files stay in temporary directories):

```bash
python -m pytest -v
```

## Limitations & Real-World Nuances

- **Capacity is not enrollment:** The simulation caps admitted applicants as a share of the evaluation pool, while real institutions set offer targets to account for yield, waitlists, and enrollment goals. The synthetic fixed-capacity rule is a simplified teaching model, not an enrollment forecast.
- **Eligibility and capacity:** Applicants outside the model's top-capacity shortlist are initially ineligible; human review can change eligibility only for escalated applications. The final selection re-ranks eligible applicants by model score and never exceeds capacity. This makes review an override to the shortlist, not a way to increase class size.
- **Idealized human review:** `src/escalation_policy.py` models human readers with a symmetric **92%** accuracy rate across cohorts. In this simulation, reviewer decisions use `true_qualified` as an oracle; actual readers do not observe ground truth. Real admissions committees experience reader fatigue, inter-rater variance, and their own cognitive biases.
- **Synthetic context proxies:** AP offerings and work/caregiving hours are simulated signals correlated with cohort. The cohort-blind routing rule is an illustration, not evidence that these proxies or thresholds are fair, valid, or appropriate in real admissions.
- **Longitudinal experiment:** Both regimes start from the same population and use separate seeded random streams for feature generation, reviewer decisions, and environmental shocks. Review draws therefore do not shift later feature or shock draws; applicant states still diverge as decisions compound.
- **Post-hoc escalation vs. upstream feature engineering:** Even after targeted escalation, **50.0%** of qualified non-traditional applicants remain auto-rejected below `0.04`. In production, context-aware escalation should be paired with upstream feature engineering (such as replacing raw AP counts with `ap_utilization = ap_courses_taken / school_ap_cap`).

## Generated Outputs

The CSVs and charts in `outputs/` are checked-in reference artifacts for the README and notebook. Running the pipeline or notebook overwrites those files when using the default output directory. Use the runner's `--output-dir` option to generate a separate pipeline run without changing the reference artifacts.

## "Try This Next" Winter Break Challenge

1. **Test human review accuracy sensitivity:** Edit `human_acc` in `walkthrough.ipynb` (Section 5) or `src/escalation_policy.py` (e.g., `0.85` vs. `0.95`)—at what human review accuracy does escalating borderline cases stop improving overall accuracy?
2. **Hidden Confounder / Omitted Variable Experiment:** Add an unmeasured **legacy / institutional-familiarity boost** into `historical_admit` without exposing it to the model. How does that shift the decision boundary and change which applicants land inside or outside the review band?
3. **Capacity-Constrained Escalation:** Suppose human admissions readers can only review **5%** of total applications. Design a ranking or context-triggered rule to select which applications get human review first.

## Research Attribution

This sandbox is inspired by and anchored to the empirical priors in the systemic fairness research of **Arindam Ray, Balaji Padmanabhan, and Lina Bouayad**:
- *INFORMS Journal on Computing* simulation repository: https://github.com/INFORMSJoC/2023.0523
- *Systemic Fairness* (`arXiv:2304.06901`): https://arxiv.org/pdf/2304.06901.pdf