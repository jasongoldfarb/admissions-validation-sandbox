My oldest is a freshman in high school, which means the college application process is suddenly top of mind in our house.

With a Ph.D. in Higher Education—where I studied equity in schools—and a career in data science, I’m looking at this through three lenses. As a parent, I wonder how a young adult can be reduced to a few data points—and what we might miss. As a researcher, I know how deeply unequal access shapes a student’s record. And as a data scientist, I understand that colleges need tools to do their work effectively. The challenge is making sure those tools help without losing sight of the person behind the application.

I was poking around online and came across research by Ray, Padmanabhan, and Bouayad on systemic fairness in admissions. It got me thinking about how those ideas might play out in practice, so I built a self-contained Python sandbox to explore them: a single-cycle audit and a 50-cycle simulation of a selective admissions screener. I wanted to explore what happens when a model learns from historical admissions decisions shaped by unequal opportunity—and then those decisions influence who gets admitted in the next cycle.

Here is what stood out:

• Headline accuracy was deceptive. The baseline model hit 82.5% overall accuracy—and actually scored higher on non-traditional applicants (83.2%) than traditional ones (81.8%). Yet qualified non-traditional applicants faced a 68.5% wrongful auto-rejection rate, compared to 12.0% for traditional applicants.

• Letting the model say “I’m not sure” worked. Without changing the underlying model, I added a cohort-blind escalation rule using observable context (school AP caps and outside work/caregiving hours). Routing just 6.4% of applications to human review cut non-traditional wrongful rejections by 27% and lifted overall accuracy to 85.5%.

• Over 50 cycles, the automated baseline created a self-fulfilling prophecy. When admitted candidates gain a modest advantage each cycle (+0.05 quality boost) and the model retrains on admitted outcomes, the automated baseline’s subgroup quality gap more than tripled—even as its wrongful rejection rate looked like it was improving. Why? Because compounding bias gradually pushed qualified non-traditional candidates out of the top 20% altogether. Context-aware human review neutralized that gap and drove wrongful rejections to 0% by cycle 8.

Whether it’s college admissions, lending, or claims, the validation lesson is the same: aggregate metrics can look healthiest right when a feedback loop is locking in a blind spot.

If you want to poke at the code, break my assumptions, or test different human-review accuracy rates, the full repo and an interactive Google Colab notebook are linked in the first comment.

#DataScience #ModelValidation #MachineLearning #ResponsibleAI #Python

**First Comment (post right after publishing):**

Everything is open and runnable with zero local setup:

• GitHub Repo & Full Walkthrough: [https://github.com/jasongoldfarb/admissions-validation-sandbox](https://www.google.com/search?q=https://github.com/jasongoldfarb/admissions-validation-sandbox)
• INFORMS Journal on Computing Research Repo (Ray, Padmanabhan, & Bouayad): [https://github.com/INFORMSJoC/2023.0523](https://github.com/INFORMSJoC/2023.0523)
• Systemic Fairness Paper (arXiv): [https://arxiv.org/pdf/2304.06901.pdf](https://arxiv.org/pdf/2304.06901.pdf)
