# Analysis plan R4

Written and pushed before the R4 analyses were run. These are time-stamped analysis plans, not third-party registrations; all analyses are exploratory.

## What is already known

- All results of plans R1–R3.
- The template-only shared-source rule for code was fixed in `PREREG_R3.md` before any R3 analysis was run. It was justified there by the single reference solution. The 66% removal by the peer filter was observed afterwards, as a pre-specified variant.
- An internal check has already compared paragraphs entering as large insertions (X) with typed paragraphs (T) in their 4-gram and 5-gram overlap with the student's own replies and with other students' replies. It found no excess. Analysis F-4 below repeats this check with the R4 entry rule and is therefore **post hoc in direction**.

## E. Entry rule based on edit size (both datasets)

The R3 entry rule compared the net change in length between consecutive snapshots, so it cannot see a paste that replaces a selection of similar length. From R4 on, the entry mode of an n-gram is based on the number of **inserted tokens** between the previous snapshot and the snapshot in which the n-gram first appears. This is the sum of the "insert" and "replace" segments of a token-level alignment (Python `difflib.SequenceMatcher`, `autojunk=False`).
- The thresholds are unchanged: at most 5 inserted tokens means T in FLoRA, at most 20 in RelianceScope.
- This rule becomes the primary rule for T and X.
- Results under the net-change rule (R3) are reported alongside.

## F. FLoRA

- F-1. Validation of T/X. For the 68 students with logged paste events, report the share of T tokens and the share of X tokens that are covered by 8-grams of their logged pasted text, under both entry rules.
- F-2. Class shares: mean and median of each class, with 2,000-resample bootstrap intervals.
- F-3. Paragraph classes (typed: at least 90% T; untraced: at least 90% X), recomputed under the edit-size rule. Re-run the R3 detector analyses (AUC chatbot vs typed, chatbot vs untraced, untraced vs typed; first-language difference within typed; clustered regression) on the existing detector scores.
- F-4 (post hoc in direction). Mean 4-gram and 5-gram containment of typed and untraced paragraphs in (a) the student's own replies and (b) the replies of 20 randomly drawn other students.

## R. RelianceScope

- R-1. Chance baseline (V3 analogue). For each student, compute the coverage of the final code by 8-grams of each other student's replies (78 nulls), excluding template n-grams. Report the share of students whose U* exceeds their own null 99th percentile, and chance-corrected U*, defined as U* minus the null median.
- R-2. U* excluding n-grams of the reference solution given to the chatbot, that is, C text that convergence on the correct answer cannot explain.
- R-3. Conservative U*: C n-grams that also occur in the student's logged copied strings.
- R-4. Human-code validation with length controls. Primary positive class: "copy the response". Secondary: "copy" or "write over the response". For each, report:
  - the AUC of reply length (tokens) alone;
  - the AUC of code tokens in the replies alone (tokens inside fenced code blocks) alone;
  - the AUC of segment uptake normalised by the number of reply 8-grams;
  - the AUC of raw segment uptake;
  - the incremental AUC of adding log(1 + uptake) to a logistic model with log(1 + reply tokens) and log(1 + code tokens). The models use out-of-fold predictions from student-grouped five-fold cross-validation with 20 random fold assignments; the interval comes from 2,000 bootstrap resamples of students.
- R-5. Entry rule as in E: class shares (mean, median, interval).
- R-6. Post-test:
  - Tobit model right-censored at 10, with pre-test and standardised U*.
  - Sensitivity: ordered logit with the same predictors.
- R-7. Report the number of students with at least one coded segment, which explains n = 69 against 79.

## Not done

No other models, thresholds or criteria after seeing R4 results.

## Deviations and additions

R4-1 (post hoc refinement of F-1). Logged pastes also contain text that the student typed and later moved within the document. "Covered by a logged paste" is therefore not the same as "entered by a paste". `src/r4_flora_tx_timing.py` adds a stricter test: at the n-gram level, it asks whether a logged paste containing the n-gram occurred within 10 s of the n-gram's first editor appearance. Both versions are reported.

R4-2. In R-4, segment uptake counts only n-grams in class C, so echo n-grams are excluded. The R3 count did not exclude them. As a result, the AUC for the merged positive class is 0.746 here, against 0.759 in R3.

## Outcome (added after the analyses)

Results are in `results/r4_rs_stats.json`, `results/r4_flora_stats.json` and `results/r4_flora_tx_timing.json`.
