# Analysis plan, Study 1 (FLoRA): measured uptake of chatbot text

Written and committed before any association between dialogue behaviour, uptake and outcomes was computed. Before writing it we inspected only file structure, row counts, the number of students (275), the distribution of proposal scores (0–15) and proposal length. These are time-stamped analysis plans, not registrations with a third party; all analyses are exploratory.

## Data

FLoRA Open Data IPS (Li et al., 2026; Figshare 10.6084/m9.figshare.32321499; CC BY 4.0). Files used: 5.1 dialogue data, 5.3 writing log, 5.4 proposal and score, 5.5 biographic survey, 5.6 prior-knowledge survey, 5.7 post-task survey. Unit of analysis: the student (n = 275 with a proposal and at least one dialogue turn).

## Measures

**Uptake U (primary label).** Tokenise the final proposal and all chatbot responses to the same student (lower case, alphanumeric tokens). U is the share of proposal tokens covered by at least one word 8-gram that also occurs in that student's chatbot responses. Sensitivity: n = 5 and n = 12.

**Null calibration.** For each student, recompute U against the chatbot responses of a randomly chosen other student (200 draws per student). The 99th percentile of this null distribution defines the overlap expected from generic phrasing alone. We report U and the excess U − null median.

**Binary label.** High uptake = top quartile of U across students. Sensitivity: U above the student-pooled null 99th percentile.

**Exposure (baseline).** Total number of chatbot words received; number of student turns.

**Dialogue behaviour.** Shares of the student-utterance codes in the release (Idea exploration: Knowledge, Judgement, Preference, Provide Context, Revise, Inspire; Task Clarification; Writing Advice; Interpret; Chitchat; Others; codes with fewer than 20 occurrences pooled into Other), plus log number of turns.

**Covariates.** Prior knowledge in data science, years of experience, prior GenAI familiarity, mother tongue English (yes/no), degree level, proposal length in words.

**Outcome.** Proposal score (0–15, five rubric criteria). Secondary: post-task self-reported understanding improvement.

## Research questions and decision rules

RQ1 (measurement). Distribution of U and of excess U; share of students with U above the null 99th percentile. Descriptive.

RQ2 (detection beyond exposure). Logistic regressions for high uptake: (a) exposure only, (b) behaviour shares + log turns, (c) exposure + behaviour. Five-fold stratified CV repeated 20 times with different seeds; out-of-fold AUC averaged over repeats. Incremental AUC (c) − (a) with a 95% student-level bootstrap interval (2,000 resamples of out-of-fold predictions). Behaviour adds information if the lower bound exceeds 0; the gain is negligible if the upper bound is below 0.02 (smallest gain of interest); otherwise inconclusive. Coefficient signs with 95% bootstrap intervals from model (c) fitted on all students, inputs standardised. Early detection: repeat (b) and (c) using only each student's first half of turns.

RQ3 (uptake and the product score). OLS of proposal score on U (standardised) with covariates; HC3 standard errors; two-sided. Report the coefficient and 95% CI. Secondary: Spearman correlation of U with self-reported understanding improvement among survey respondents. No causal interpretation.

RQ4 (equity). Mean U and the flag rate of model (c) by mother tongue (English / other), with bootstrap intervals for the difference.

## Exclusions and missing data

Students with no chatbot turns are excluded (none expected in the release). Covariates missing for a student are imputed with the median (numeric) or a "missing" category, and the number imputed is reported.

## Not done

No hyperparameter search beyond L2-regularised logistic regression with C = 1. No outcome-based selection of n-gram length, thresholds or codes.
