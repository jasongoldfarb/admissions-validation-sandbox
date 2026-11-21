# Selective Admissions Model Validation & Escalation Sandbox

A self-contained Python model-validation sandbox exploring **why strong headline accuracy can hide subgroup blind spots**â€”and how a **human-in-the-loop review band** ("letting the model say *I'm not sure*") reduces wrongful auto-rejections on applicants whose qualitative context gets flattened by quantitative snapshots.

Built as a low-friction **winter-break / New Year's coding sandbox** for data scientists, analysts, and curious builders.

## Key Findings (Simulated Holdout, N = 3,000)

- **Headline accuracy looks solid:** The baseline screener trained on biased historical admissions data reaches **87.2% overall accuracy** (**89.4%** on traditional applicants, with a **12.3%** wrongful auto-rejection rate).
- **Subgroup blind spot:** For qualified **non-traditional applicants** (such as students with fewer AP courses offered at their school, heavy outside work/caregiving hours, or upward grade trajectories), the wrongful auto-rejection rate (false negative rate) reaches **67.8%**.
- **Flattened context:** Many qualified non-traditional applicants get pushed into low predicted-probability buckets (below 0.20) because raw counts (like ATs taken or extracurriculars) miss school caps, outside commitments, and late momentum.
- **Targeted human escalation:** Pairing a standard [0.20, 0.65] review band for traditional applicants with a lower floor [0.04, 0.65] for non-traditional applicants (assuming **92%** human review accuracy) escalates just **4.9%** of applications (148 of 3,000), cutting non-traditional wrongful auto-rejections from **67.8% to 55.6%** (an **18.0% relative drop**) and lifting overall accuracy from **87.2% to 89.2%** without changing the underlying scoring model.

## Charts

![Aggregate vs. Subgroup Performance](outputs/chart1_aggregate_vs_subgroup.png)

![Human-in-the-Loop Trade-Off Curve](outputs/chart2_escalation_tradeoff.png)

## Quickstart

```bash
git clone https://github.com/jasongoldfarb/admissions-validation-sandbox.git
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
â”œâ„€â„€ src/
â”œâ„€â„€   â”œâ„€â„€ baseline_repro.py       # Session 1: Baseline simulation & subgroup metrics
â”œâ„€â„€   â”œâ„€â„€ audit_borderline.py     # Session 2: Score-zone slice & flattened-context audit
â”œâ„€â„€   â”œâ„€â„€ escalation_policy.py    # Session 3: Review band + human_review_accuracy simulation
â”œâ„€â„€   â”œâ„€â„€ make_charts.py          # Session 4: Generate PNG visuals from simulation outputs
â”œâ„€â„€ outputs/                    # Generated CSVs and PNG charts
â”œâ„€â„€ requirements.txt
â”œâ„€â„€I5¹µ)€((ŒŒ€‰QÉäQ¡¥Ì9•áĞˆ]¥¹Ñ•È	É•…¬¡…±±•¹”((Ä¸€¨©Q•ÍĞ¡Õµ…¸É•Ù¥•Ü…ÕÉ…äÍ•¹Í¥Ñ¥Ù¥Ñäè¨¨‘¥Ğ¡Õµ…¹}…€¥¸ÍÉŒ½•Í…±…Ñ¥½¹}Á½±¥ä¹Áå€€¡”¹œ¸°€À¸àÕ€ÙÌ¸€À¸äÕ€§ŠQ…Ğİ¡…Ğ¡Õµ…¸É•Ù¥•Ü…ÕÉ…ä‘½•Ì•Í…±…Ñ¥¹œ‰½É‘•É±¥¹”…Í•ÌÍÑ½À¥µÁÉ½Ù¥¹œ½Ù•É…±°…ÕÉ…äü(È¸€¨©!¥‘‘•¸½¹™½Õ¹‘•È€¼=µ¥ÑÑ•Y…É¥…‰±”áÁ•É¥µ•¹Ğè¨¨‘…¸Õ¹µ•…ÍÕÉ•€¨©±•…ä€¼¥¹ÍÑ¥ÑÕÑ¥½¹…°µ™…µ¥±¥…É¥Ñä‰½½ÍĞ¨¨¥¹Ñ¼¡¥ÍÑ½É¥…±}…‘µ¥Ñ€İ¥Ñ¡½ÕĞ•áÁ½Í¥¹œ¥ĞÑ¼Ñ¡”µ½‘•°¸!½Ü‘½•ÌÑ¡…ĞÍ¡¥™ĞÑ¡”‘•¥Í¥½¸‰½Õ¹‘…Éä…¹¡…¹”İ¡¥ …ÁÁ±¥…¹ÑÌ±…¹¥¹Í¥‘”½È½ÕÑÍ¥‘”Ñ¡”É•Ù¥•Ü‰…¹ü(Ì¸€¨©…Á…¥Ñäµ½¹ÍÑÉ…¥¹•Í…±…Ñ¥½¸è¨¨MÕÁÁ½Í”¡Õµ…¸…‘µ¥ÍÍ¥½¹ÌÉ•…‘•ÉÌ…¸½¹±äÉ•Ù¥•Ü€¨¨Ô”¨¨½˜Ñ½Ñ…°…ÁÁ±¥…Ñ¥½¹Ì¸•Í¥¸„É…¹­¥¹œ½È½¹Ñ•áĞµÑÉ¥•É•ÉÕ±”Ñ¼Á¥¬€©İ¡¥ ¨€Ô”•Ğ¡Õµ…¸•å•Ì™¥ÉÍĞ¸((ŒŒI•Í•…É ÑÑÉ¥‰ÕÑ¥½¸()Q¡¥ÌÍ…¹‘‰½à¥Ì¥¹ÍÁ¥É•‰ä…¹‰Õ¥±‘ÌÕÁ½¸Ñ¡”ÍåÍÑ•µ¥Œ™…¥É¹•ÍÌÉ•Í•…É …¹½Á•¸µÍ½ÕÉ”Í¥µÕ±…Ñ¥½¸İ½É¬½˜€¨©É¥¹‘…´I…ä°	…±…©¤A…‘µ…¹…‰¡…¸°…¹1¥¹„	½Õ…å…¨¨è(´€©%9=I5L)½ÕÉ¹…°½¸½µÁÕÑ¥¹œ¨Í¥µÕ±…Ñ¥½¸É•Á½Í¥Ñ½Éäè¡ÑÑÁÌè¼½¥Ñ¡Õˆ¹½´½%9=I5M)½¼ÈÀÈÌ¸ÀÔÈÌ(´€©MåÍÑ•µ¥Œ…¥É¹•ÍÌ¨€¡…Éa¥ØèÈÌÀĞ¸ÀØäÀÅ€¤è¡ÑÑÁÌè¼½…Éá¥Ø¹½Éœ½Á‘˜¼ÈÌÀĞ¸ÀØäÀÄ¹Á‘˜(