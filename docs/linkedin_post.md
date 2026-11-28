My oldest is a freshman in high school, which means the college application process is suddenly top of mind in our house.

It’s a weird spot to sit in. As a parent, I wonder how a young adult gets reduced to a few data points—and what gets missed. With a Ph.D. in Higher Education, where I studied equity and college admissions, I know how unequal access shapes a student’s opportunities. And as a data scientist, I understand that colleges need tools to manage large application volumes, including applications submitted through platforms like the Common App. The hard part is making sure those tools help without losing sight of the person behind the application.

Over Thanksgiving, I started thinking about how admissions decisions are made, and what happens when models learn from historical decisions shaped by unequal opportunity. I started digging into the research and came across work by Ray, Padmanabhan, and Bouayad on systemic fairness in admissions. Inspired by their research, I built a self-contained Python sandbox to explore a fixed-capacity admissions scenario (20% capacity by default) and test how a context-aware human-review rule affects screening outcomes. It includes both a single-cycle audit and a 50-cycle simulation.

A few things stood out:

- Headline accuracy was deceptive. The baseline model hit 82.5% overall accuracy, and actually scored higher on non-traditional applicants (83.4%) than traditional ones (81.7%). Yet qualified non-traditional applicants faced a 67.6% wrongful auto-rejection rate, compared to 11.6% for traditional applicants.
- Letting the model say “I’m not sure” helped. Keeping the same model scores, I added a cohort-blind escalation rule using observable context, such as school AP caps and outside work/caregiving hours. Routing 6.4% of applications (193 of 3,000) to review raised overall accuracy from 82.5% to 84.3% and lowered the non-traditional wrongful auto-rejection rate from 67.6% to 61.9% (a 5.7 percentage-point drop). Reviewer decisions in this simulation use a hidden readiness label and assume 92% accuracy, which is an intentionally idealized proxy rather than information real reviewers would have.
- Falling wrongful-rejection rates aren’t the whole story. In the 50-cycle simulation, they declined in both regimes, while the traditional-minus-non-traditional simulated mean-quality gap crossed zero and ended negative. In the model, admitted applicants who meet the readiness threshold gain +0.05 quality each cycle, while those admitted below it lose 0.05; the screener also retrains on admissions outcomes. Non-traditional wrongful auto-rejections fell from 63.3% to 3.5% in the baseline and from 55.8% to 2.5% with context-aware review. Meanwhile, the simulated mean-quality gap shifted from +0.0134 to -0.0302 in the baseline and -0.0370 with review (a negative gap means simulated mean quality for non-traditional applicants ended up higher). These results are illustrative and depend heavily on the simulation’s assumptions, so they don’t show that either policy eliminates bias.

The takeaway extends beyond admissions. As data science leaders, especially those of us mentoring newer colleagues, we have a responsibility to look beyond aggregate accuracy. That means auditing who gets missed in the margins, making room for human context in consequential decisions, and testing how today’s choices reshape tomorrow’s pool.

If you want to experiment with the code, challenge my assumptions, or test different human-review accuracy rates, the full repo and an interactive Google Colab notebook are linked in the first comment.

#DataScience #ModelValidation #MachineLearning #ResponsibleAI #Python

## Pinned First Comment
Everything is open and runnable with zero local setup:

- GitHub Repo & Full Walkthrough: [https://github.com/jasongoldfarb/admissions-validation-sandbox](https://github.com/jasongoldfarb/admissions-validation-sandbox)
- INFORMS Journal on Computing Research Repo (Ray, Padmanabhan, & Bouayad): [https://github.com/INFORMSJoC/2023.0523](https://github.com/INFORMSJoC/2023.0523)
- Systemic Fairness Paper (arXiv): [https://arxiv.org/pdf/2304.06901.pdf](https://arxiv.org/pdf/2304.06901.pdf)