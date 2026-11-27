# LinkedIn Post (December Release)

My oldest is a freshman in high school, which means the college application process is suddenly top of mind in our house.

I’m looking at this through three lenses. As a parent, I wonder how a young adult can be reduced to a few data points—and what we might miss. With a Ph.D. in Higher Education studying equity in schools, I know how deeply unequal access shapes a student’s record. And as a data scientist, I understand that colleges need tools to do their work effectively. The challenge is making sure those tools help without losing sight of the person behind the application.

Over Thanksgiving, we were talking about college majors and schools, and it got me thinking about how admissions decisions are made—and what happens when models learn from historical decisions shaped by unequal opportunity. I started looking into the research and came across work by Ray, Padmanabhan, and Bouayad on systemic fairness in admissions. Building on their framework, I created a self-contained Python sandbox that adds two practical layers: a fixed 20% seat cap to reflect real enrollment targets, and a context-aware human-review step designed to make screening more equitable—running both a single-cycle audit and a 50-cycle simulation.

Here is what stood out:

- Headline accuracy was deceptive. The baseline model hit 82.5% overall accuracy—and actually scored higher on non-traditional applicants (83.4%) than traditional ones (81.7%). Yet qualified non-traditional applicants faced a 67.6% wrongful auto-rejection rate, compared to 11.6% for traditional applicants.

- Letting the model say “I’m not sure” helped in this simulation. Keeping the same model scores, I added a cohort-blind escalation rule using observable context (such as school AP caps and outside work/caregiving hours—two example parameters that could easily be expanded). Routing 6.4% of applications (192 of 3,000) to human review raised overall accuracy from 82.5% to 84.2% and lowered the non-traditional wrongful auto-rejection rate from 67.6% to 61.9%—a 5.7 percentage-point decrease.

- The 50-cycle simulation showed why falling wrongful-rejection rates aren’t the whole story: they declined in both regimes, while the traditional-minus-non-traditional quality gap crossed zero and ended negative. In the model, admitted applicants who meet the readiness threshold gain +0.05 quality each cycle, while those admitted below that threshold lose 0.05; the screener also retrains on admissions outcomes. Non-traditional wrongful auto-rejections fell from 63.3% to 3.5% in the baseline and from 55.8% to 2.5% with context-aware review. Meanwhile, the quality gap shifted from +0.0134 to -0.0302 in the baseline and -0.0370 with review (a negative gap means non-traditional mean quality ended up higher). These results are illustrative and depend on the simulation’s assumptions, so they don’t show that either policy eliminates bias. But the validation approach is broadly useful: make assumptions explicit, audit outcomes across groups, and test how decisions shape the next round of data.

Whether it’s college admissions, lending, or claims, the validation takeaway is practical: stop signing off on models using aggregate accuracy alone. Audit who gets missed in the margins, build room for human context when the data is thin, and test how today’s decisions reshape tomorrow’s pool.

If you want to poke at the code, break my assumptions, or test different human-review accuracy rates, the full repo and an interactive Google Colab notebook are linked in the first comment.

#DataScience #ModelValidation #MachineLearning #ResponsibleAI #Python

---

## Pinned First Comment

Everything is open and runnable with zero local setup:

- GitHub Repo & Full Walkthrough: https://github.com/jasongoldfarb/admissions-validation-sandbox
- INFORMS Journal on Computing Research Repo (Ray, Padmanabhan, & Bouayad): https://github.com/INFORMSJoC/2023.0523
- Systemic Fairness Paper (arXiv): https://arxiv.org/pdf/2304.06901.pdf

(Tip: Click the "Open in Colab" badge at the top of the README if you want to run the sensitivity sandbox right in your browser.)