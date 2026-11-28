My oldest is a freshman in high school, which means the college application process is suddenly top of mind in our house.

It's a weird spot to sit in. As a parent, I wonder how a young adult gets reduced to a few data points, and what gets missed. With a Ph.D. in Higher Education, where I studied equity and college admissions, I know how unequal access shapes a student's opportunities. And as a data scientist, I understand that colleges need tools to manage large application volumes, including applications submitted through platforms like the Common App. The hard part is making sure those tools help without losing sight of the person behind the application.

Over Thanksgiving, I started thinking about how admissions decisions are made, and what happens when models learn from historical decisions shaped by unequal opportunity. I dug into the research and came across work by Ray, Padmanabhan, and Bouayad on systemic fairness in admissions. Inspired by their research, I built a self-contained Python sandbox to explore a fixed-capacity admissions scenario (20% capacity by default) and test how a context-aware human-review rule affects screening outcomes. It includes both a single-cycle audit and a 50-cycle simulation.

A few things stood out:

- **Headline accuracy was deceptive.** The baseline model hit 82.5% overall accuracy and scored slightly higher on non-traditional applicants (83.4%) than traditional ones (81.7%). Yet the wrongful auto-rejection rate (qualified applicants rejected without human review) was 67.6% for non-traditional applicants versus 11.6% for traditional ones. Accuracy can look fine when most of a group is correctly rejected, even if the qualified few are missed.
- **Letting the model say "I'm not sure" helped, but not enough.** Keeping the same model scores, I added an escalation rule that uses context, not group labels, such as school AP caps and outside work or caregiving hours. Routing 6.4% of applications (193 of 3,000) to human review raised overall accuracy from 82.5% to 84.3% and lowered the non-traditional wrongful auto-rejection rate from 67.6% to 61.9%. That is still far too high. Reviewers here are assumed to be 92% accurate using a hidden readiness label, an idealized proxy for what real reviewers would know.
- **Fixing today's error rate doesn't settle tomorrow's pool.** Over 50 cycles, non-traditional wrongful auto-rejections fell sharply in both setups (63.3% to 3.5% in the baseline, 55.8% to 2.5% with review), while the simulated quality gap between the two groups flipped sign. These dynamics depend heavily on my assumptions (admitted applicants gain or lose 0.05 quality per cycle depending on readiness, and the screener retrains on its own outcomes), so treat them as illustrative, not evidence that either policy removes bias.

Honestly, the result that stuck with me most was the one that looked best on paper: higher accuracy for the very group the model was failing the most.

I keep coming back to the same thought as both a parent and a data scientist. If you lead or mentor data teams, look past aggregate accuracy. Ask who gets missed at the margins, where human context belongs in high-stakes decisions, and how today's choices reshape tomorrow's pool.

If you want to experiment with the code, challenge my assumptions, or test different human-review accuracy rates, the full repo and an interactive Google Colab notebook are linked in the first comment.

#DataScience #ModelValidation #MachineLearning #ResponsibleAI #Python

## Pinned First Comment
Everything is open and runnable with zero local setup:

- GitHub Repo & Full Walkthrough: [https://github.com/jasongoldfarb/admissions-validation-sandbox](https://github.com/jasongoldfarb/admissions-validation-sandbox)
- INFORMS Journal on Computing Research Repo (Ray, Padmanabhan, & Bouayad): [https://github.com/INFORMSJoC/2023.0523](https://github.com/INFORMSJoC/2023.0523)
- Systemic Fairness Paper (arXiv): [https://arxiv.org/pdf/2304.06901.pdf](https://arxiv.org/pdf/2304.06901.pdf)