# Analysis plan R3

Written and pushed before the R3 analyses of the second dataset were run. These are time-stamped analysis plans, not third-party registrations; all analyses are exploratory.

## What was known before this plan

- FLoRA results of plans R1 and R2 (`results/r1_stats.json`, `results/r2_stats.json`).
- An internal reanalysis of the R2 paragraph labels showed that most paragraphs labelled "composed" in R2 did not enter the editor through typing. Most of them arrived as large insertions that the logs cannot trace to any source. In that reanalysis, the first-language difference in detector scores was not reliable among typed paragraphs. The FLoRA analyses in part F below are therefore **post hoc**: their direction is already known, and they are reported as such.
- For the second dataset (RelianceScope; Jin et al., 2026; 79 students, Vue.js programming activity with a tutoring chatbot), we inspected only the structure: table fields and ID linkage, the number of edit snapshots and their spacing (median 15 s), the counts of human response-use codes, and the descriptive distribution of the pre-test, post-test and perceived-reliance scores (post-test median 10 of 10). No association between any process measure and any outcome or code was computed.

## P. Problem definition (fixed now)

Each token of a student's final artefact is assigned to one of five provenance classes, using the dialogue, the artefact snapshots and the copy/paste events:

- C, chatbot-first: covered by an n-gram that occurs in a chatbot reply, was not supplied by the student earlier (in a prompt or in the artefact), is not from a shared source, and appears in the artefact no earlier than its first reply (δ = 0).
- E, echo: covered by an n-gram that the student supplied before the chatbot produced it.
- S, shared source: covered by an n-gram from a source available to all students. For FLoRA these are reading highlights, or the n-gram occurs in at least five other proposals. For RelianceScope this is the starter template.
- T, typed: not C, E or S, and entered through small edits.
- X, untraced insertion: not C, E or S, and entered as a large insertion.

The rules apply in this order: C, E, S, then T or X by entry mode. Uptake U* is the share of class C, as before. The new classes T and X separate text composed in the editor from text that arrived from outside the logs.

Entry mode. An n-gram's entry is the first snapshot that contains it.
- FLoRA (snapshots per keystroke): the n-gram is typed if the net change at entry is at most 5 tokens. A token is T if at least one covering n-gram was typed; otherwise it is X.
- RelianceScope (snapshots about every 15 s): the threshold is a net change of at most 20 tokens, the same as the jump rule of R1. This is coarser, and it is reported as such.

## R. RelianceScope (second dataset, code)

Tokens: alphanumeric runs of the code, as for prose (`[a-z0-9]+` after lower-casing); n = 8. The final artefact is the last edit snapshot of the student's conversation.

Shared source: n-grams of the starter template, which is the earliest snapshot common to all students. The peer filter (at least five other students) is **not** part of the primary measure for code: the task has one reference solution, which the chatbot was given, so correct code converges across students whatever its origin. The peer-filtered version is reported as a variant, to show how much a prose-style filter removes in a single-solution task.

Echo: n-grams present in the student's earlier messages, or in the code snapshot attached to the student's messages, before the reply that contains them.

R-a. Distribution of the provenance classes: median and mean U*, shares ≥ 25% and ≥ 50%, with bootstrap intervals; n = 5 and n = 12 as sensitivity.

R-b. Copy events: the share of copied strings (logged "copied strings") that occur in an earlier chatbot reply (at least one 8-gram, or the whole string if it has fewer than 8 tokens); and the share of C tokens covered by n-grams of copied strings.

R-c. Human criterion. Segments are taken from the human-coded chat log; a segment starts at a coded row. Segment uptake is the number of tokens that become C between the segment's start and the next segment's start (or the final snapshot), counted from n-grams of replies in that segment.
- Positive class: Response-Use (detailed) is "copy the response" or "write over the response". Both reproduce the chatbot's text.
- Negative class: all other codes.
- Report the AUC of segment uptake, and the share of segments with any uptake by detailed code. Intervals come from 2,000 bootstrap resamples of students.
- Secondary: among segments with uptake, compare the share of uptake tokens entered as large insertions (X-like entry) between "copy the response" and "write over the response".

R-d. Student level: Spearman correlations of U* with the share of segments coded "copy the response" or "write over the response", with the share coded passive response use, with the share coded passive help-seeking, and with perceived reliance (Likert).

R-e. Unaided learning outcome: OLS of post-test on pre-test and standardised U*, with HC3 standard errors. Reported alongside: the share at ceiling (10/10), and a logistic regression of reaching 10/10 on pre-test and U*.

## F. FLoRA provenance classes and detectors (post hoc)

F-a. Shares of T and X among the composed paragraphs of R2, and the shares of T, X and C tokens over all proposals. For students with and without trace logs, report separately.

F-b. Typed paragraphs: at least 90% T tokens. Untraced paragraphs: at least 90% X tokens. For D1 to D3, report the AUC of chatbot vs typed, chatbot vs untraced, and untraced vs typed.

F-c. First-language difference in detector scores within typed and within untraced paragraphs. Also report a regression of the standardised score on first language, log length and untraced status, with standard errors clustered by student.

F-d. D3 sensitivity: Binoculars with the roles of observer and performer in the perplexity term swapped (the notation of the paper rather than of its released code, which we follow).

## Not done

No further datasets, detectors, thresholds or outcome models after seeing R3 results.
