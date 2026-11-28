# Source Modules (`src/`)

Pipeline modules for the Selective Admissions Model Validation & Escalation Sandbox. Run all stages with `python src/run_pipeline.py`; individual scripts can also be run directly. Outputs default to the repository's `outputs/` directory even when invoked from another working directory.

This project is inspired by Ray, Padmanabhan, and Bouayad's systemic fairness research and uses selected empirical priors from that work; it does not reproduce the paper's multi-university simulation. The original code models applications across universities and tiers, SAT screening, and noisy weighted evaluations. This sandbox instead models a single capacity-constrained pool with a logistic-regression screener and context-aware review.

The single-cycle and longitudinal simulations share the readiness scorer in `baseline_repro.py`: latent academic and non-academic quality, AP utilization, work/care hours, upward grade trajectory, and seeded noise. Qualification is assigned to the top 20% of each simulated pool or cycle.

Admissions are capacity-constrained separately from that readiness label. The default target is 20% of the applicant pool. Model probabilities rank applicants; the baseline shortlists the top capacity-sized group. HITL can change eligibility for escalated applicants before a final score ranking selects up to the seat cap. Score ties retain input order; if too few applicants remain eligible, seats are left unfilled.

## End-to-End Runner

`run_pipeline.py` runs the baseline, audit, escalation, longitudinal simulation, and chart generation in order. Use `python src/run_pipeline.py --output-dir /path/to/run` to write artifacts to a separate directory.

---

## 1. `baseline_repro.py`

### Purpose & Pipeline Role
Generates the simulated applicant pool using selected empirical priors from Ray, Padmanabhan, and Bouayad (*INFORMS Journal on Computing*, `INFORMSJoC/2023.0523`), trains a baseline logistic regression screener on synthetic historical admissions labels, and evaluates overall vs. cohort-level performance against ground-truth readiness. This is a new single-institution training model, not an implementation of the paper's multi-university algorithm.

- **Inputs:** None (self-contained generator with fixed seed `42`).
- **Outputs:** `outputs/baseline_applicants.csv` (3,000-row stratified holdout set with `pred_prob`, initial `model_eligible` shortlist, and capacity-limited `pred_admit`).

### Key Functions & Parameters
- `generate_applicants(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame`
  Simulates applicant features, flattened qualitative context (`school_ap_cap`, `outside_hours_wk`, `upward_trajectory`), `true_qualified`, and `historical_admit`.
- `compute_true_readiness_score(...) -> np.ndarray`
  Shared readiness scoring function used by both simulations to keep the ground-truth definition aligned.
- `summarize_metrics(df: pd.DataFrame, label_col: str = "true_qualified", pred_col: str = "pred_admit") -> dict`
  Returns `n`, `accuracy`, `fnr_wrongful_reject`, `tpr_equal_opportunity`, and `missed_qualified_count`. Safely handles zero-positive edge cases without division errors.
- `main(output_dir: Optional[Path] = None, capacity_fraction: float = 0.20) -> None`
  Runs a 70/30 stratified train/test split (`random_state=42`), fits `StandardScaler` + `LogisticRegression` on the 4 raw tabular features (`gpa`, `test_score`, `ap_courses_taken`, `leadership_index`), ranks the holdout scores, shortlists the top capacity fraction, prints cohort metrics, and writes `outputs/baseline_applicants.csv`. The historical `0.50` binary cutoff is not used as the admission rule.

### Modeling & Implementation Notes
- **Empirical Priors:** `FRACTION_OF_PROTECTED = 1 - 7.8/15.4` (~49.4%), `SAT_MEAN = 1030.0`, `SAT_STD = 235.0`, `QUALITY_MU = 0.5`, `QUALITY_SIGMA = 0.1`, `TIER1_ADMIT_RATE = 0.20`.
- **Feature Bounds:** `gpa` clipped to `[2.0, 4.0]`, `test_score` rounded to nearest 10 in `[400, 1600]`, `ap_courses_taken` bounded by `[0, school_ap_cap]`, `leadership_index` clipped to `[1.0, 10.0]`.

### Standalone Usage
```bash
python src/baseline_repro.py
```

---

## 2. `audit_borderline.py`

### Purpose & Pipeline Role
Loads the holdout predictions from `baseline_repro.py`, buckets applicants into score zones, and quantifies where qualified applicants are wrongfully auto-rejected across cohorts. It also prints a side-by-side comparison of observable features vs. flattened qualitative context for missed non-traditional candidates and admitted traditional candidates.

- **Inputs:** `outputs/baseline_applicants.csv` (raises `FileNotFoundError` if missing).
- **Outputs:** `outputs/audited_applicants.csv` (adds `zone`, `ap_utilization`, and `wrongful_reject` columns).

### Key Functions & Parameters
- `assign_zone(prob: float, low: float = 0.20, high: float = 0.65) -> str`
  Maps a predicted probability to one of three ordered score zones. Zone names describe scores, not guaranteed admission decisions under capacity constraints; labels reflect custom `low` and `high` values.
- `main(output_dir: Optional[Path] = None) -> None`
  Reads `outputs/baseline_applicants.csv`, computes derived audit fields, prints the zone breakdown, cohort-by-zone missed-qualified table, and flattened-context comparison, then writes `outputs/audited_applicants.csv`.

### Modeling & Implementation Notes
- **Derived Columns:**
  - `ap_utilization = ap_courses_taken / school_ap_cap.clip(lower=1)` (guards against zero-AP schools).
  - `wrongful_reject = 1` when `true_qualified == 1` and `pred_admit == 0`, else `0`.
- **Zone Boundaries:** The borderline band `[0.20, 0.65]` is inclusive on both endpoints.

### Standalone Usage
```bash
python src/audit_borderline.py
```

---

## 3. `escalation_policy.py`

### Purpose & Pipeline Role
Simulates imperfect human-in-the-loop (HITL) review on escalated applications, sweeps six standard probability review bands from 0% to 100% escalation, and compares them with the no-escalation capacity baseline and a cohort-blind, context-triggered escalation policy.

- **Inputs:** `outputs/audited_applicants.csv` (raises `FileNotFoundError` if missing).
- **Outputs:** `outputs/escalation_tradeoff.csv` (8-row summary with review workload, capacity, eligible/admitted/unfilled seat counts, accuracy, and cohort-level FNR/TPR metrics).

### Key Functions & Parameters
- `evaluate_policy(df: pd.DataFrame, escalate_mask: pd.Series, human_review_accuracy: float = 0.92, seed: int = 42, capacity_fraction: float = 0.20) -> dict`
  Overrides shortlist eligibility on escalated rows using a simulated reviewer who identifies `true_qualified` correctly with probability `human_review_accuracy`, then selects the highest-scoring eligible applicants up to capacity. Returns workload, target capacity, initial and revised eligible counts, admitted count, unfilled seats, overall accuracy, and subgroup FNR/TPR.
- `main(output_dir: Optional[Path] = None, capacity_fraction: float = 0.20) -> None`
  Reads `outputs/audited_applicants.csv`, evaluates the no-escalation baseline, six standard probability bands, and the context-aware policy at `human_review_accuracy = 0.92` (`seed = 42`), prints the comparison tables, and writes `outputs/escalation_tradeoff.csv`.

### Modeling & Implementation Notes
- **Cohort-Blind Context Trigger:** Routes on observable constraints (`school_ap_cap <= 6` or `outside_hours_wk >= 10.0`) using a lower review floor `[0.04, 0.65]`, and `[0.20, 0.65]` for all other applicants—never referencing the `cohort` label.
- **Reproducibility:** Uses `np.random.default_rng(seed)` so each policy evaluation is deterministic.
- **Capacity:** The final admit count never exceeds the configured seat target; reviewer overrides can create an underfilled class if fewer applicants remain eligible.

### Standalone Usage
```bash
python src/escalation_policy.py
```

---

## 4. `make_charts.py`

### Purpose & Pipeline Role
Reads the audited holdout predictions and escalation trade-off summary to generate publication-ready PNG visuals for the `README.md`, `walkthrough.ipynb`, and external sharing.

- **Inputs:** `outputs/audited_applicants.csv` and `outputs/escalation_tradeoff.csv` (raises `FileNotFoundError` if either is missing).
- **Outputs:**
  - `outputs/chart1_aggregate_vs_subgroup.png` — Bar chart contrasting headline accuracy against subgroup wrongful auto-rejection rates (before and after escalation).
  - `outputs/chart2_escalation_tradeoff.png` — Two-panel Cost vs. Payoff visual pairing human review workload against accuracy and FNR improvements.

### Key Functions & Parameters
- `make_chart1(audited_df: pd.DataFrame, tradeoff_df: pd.DataFrame, out_path: Path) -> None`
  Renders and saves the aggregate vs. subgroup performance chart at 200 DPI.
- `make_chart2(audited_df: pd.DataFrame, tradeoff_df: pd.DataFrame, out_path: Path) -> None`
  Renders and saves the two-panel workload vs. fairness payoff chart at 200 DPI. Automatically resolves `overall_acc`, `overall_accuracy`, or `accuracy` from `tradeoff_df` (falling back to `audited_df` if absent).
- `make_chart3(long_df: pd.DataFrame, out_path: Path) -> None`
  Renders the 50-cycle quality-gap and non-traditional FNR comparison.
- `main(output_dir: Optional[Path] = None) -> None`
  Validates upstream CSVs, ensures the output directory exists, and generates chart 1 and chart 2. Generates chart 3 when `longitudinal_dynamics.csv` is present.

### Modeling & Implementation Notes
- **Style Compatibility:** `_set_style()` selects `seaborn-v0_8-whitegrid` when available and falls back to `seaborn-whitegrid` for older Matplotlib environments.
- **Memory Safety:** Both chart functions call `plt.close(fig)` after saving so repeated calls (e.g., in notebooks or tests) do not leak figure handles.

### Standalone Usage
```bash
python src/make_charts.py
```

---

## 5. `longitudinal_sim.py`

### Purpose & Pipeline Role
Adds this sandbox's own 50-cycle experiment comparing automated selection and context-aware human review, updating latent quality and retraining the screener on observed outcomes. It uses a 1,000-person pool and quality updates of `+0.05` for admitted and qualified applicants and `-0.05` for admitted and unqualified applicants. These are parameters of this added experiment; the original repository's scenario runs across 50 random seeds are not an evolving 50-cycle process.

- **Inputs:** None; initializes the population from a fixed seed.
- **Outputs:** `outputs/longitudinal_dynamics.csv` (100 rows: 50 cycles for each policy).

### Modeling & Implementation Notes
- Uses the same readiness scorer and context signals as the single-cycle baseline.
- By default, each cycle has capacity for 20% of applicants. The highest model scores determine the initial shortlist; human review updates eligibility for escalated applicants before final score ranking and seat allocation.
- Uses independent seeded random streams for feature generation, human review decisions, and environmental shocks. Both regimes share the feature/shock seeds while review randomness does not perturb those streams.
- Human review is an oracle-accuracy abstraction: it uses the synthetic `true_qualified` label to simulate a specified probability of correctness. This is intentionally idealized and not a model of real reviewers' information.

---

## 6. `paths.py`

Provides repository-relative default output handling and allows callers to supply an output directory.

---

## 7. `run_pipeline.py`

Runs every stage in dependency order and generates all CSV and chart artifacts. The script can be invoked from any working directory:

```bash
python /path/to/admissions-validation-sandbox/src/run_pipeline.py --output-dir /path/to/run
```