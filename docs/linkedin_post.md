My oldest is a freshman in high school, which means the college application process is suddenly top of mind in our house.

With a Ph.D. in Higher Education—where I studied equity in schools—and a career in data science, I’m looking at this through three lenses. As a parent, I wonder how a young adult can be reduced to a few data points—and what we might miss. As a researcher, I know how deeply unequal access shapes a student’s record. And as a data scientist, I understand that colleges need tools to do their work effectively. The challenge is making sure those tools help without losing sight of the person behind the application.

Over Thanksgiving, we were talking about college majors and schools, and it got me thinking about how admissions decisions are made—and what happens when models learn from historical decisions shaped by unequal opportunity. I started looking into the research and came across work by Ray, Padmanabhan, and Bouayad on systemic fairness in admissions. I built a self-contained Python sandbox to explore those ideas: a single-cycle audit and a 50-cycle simulation of a selective admissions screener.

Here is what stood out:

• Headline accuracy was deceptive. The baseline model hit 82.5% overall accuracy—and actually scored higher on non-traditional applicants (83.4%) than traditional ones (81.7%). Yet qualified non-traditional applicants faced a 67.6% wrongful auto-rejection rate, compared to 11.6% for traditional applicants.

• Letting the model say “I’m not sure” helped in this simulation. Keeping the same model scores, I added a cohort-blind escalation rule using observable context (school AP caps and outside work/caregiving hours). Routing 6.4% of applications (192 of 3,000) to human review raised overall accuracy from 82.5% to 84.2% and lowered the non-traditional wrongful auto-rejection rate from 67.6% to 61.9%—a 5.7 percentage-point decrease.

• The 50-cycle simulation highlighted how feedback dynamics can complicate the story. In the model, admitted and qualified applicants gain +0.05 quality each cycle, while admitted but unqualified applicants lose 0.05; the screener also retrains on admissions outcomes. Non-traditional wrongful auto-rejections fell from 63.3% to 3.5% in the baseline and from 55.8% to 0.8% with context-aware review. Meanwhile, the traditional-minus-non-traditional quality gap shifted from +0.0134 to -0.0302 in the baseline and -0.0371 with review (a negative gap means non-traditional mean quality ended up higher). These are illustrative outcomes under the simulation’s assumptions—not evidence that either policy eliminates bias or that a falling rejection rate, on its own, means the system is fair.

Whether it’s college admissions, lending, or claims, the validation lesson is the same: a healthy headline metric can hide subgroup blind spots, and a falling error rate over time doesn’t by itself tell you whether the system is fair.

If you want to poke at the code, break my assumptions, or test different human-review accuracy rates, the full repo and an interactive Google Colab notebook are linked in the first comment.

#DataScience #ModelValidation #MachineLearning #ResponsibleAI #Python

**First Comment (post right after publishing):**

Everything is open and runnable with zero local setup:

• GitHub Repo & Full Walkthrough: [https://github.com/jasongoldfarb/admissions-validation-sandbox](https://www.google.com/search?q=https://github.com/jasongoldfarb/admissions-validation-sandbox)

• INFORMS Journal on Computing Research Repo (Ray, Padmanabhan, & Bouayad): [https://github.com/INFORMSJoC/2023.0523](https://github.com/INFORMSJoC/2023.0523)

• Systemic Fairness Paper (arXiv): [https://arxiv.org/pdf/2304.06901.pdf](https://arxiv.org/pdf/2304.06901.pdf)
