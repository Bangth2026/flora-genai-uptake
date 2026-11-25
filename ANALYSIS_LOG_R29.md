# Analysis log entry R29

Written on 25 November 2026, after the analyses below ran, for the fifteenth version of the manuscript, in response to a review of the fourteenth. It completes the correction of R28 (the system trace read from all eight sheets; DATA_FULL as in R28). Nothing in the attribution, its rules or its parameters changed.

## F-68 the lower end with tau = 2 (src/r29_flora.py lower)

C tokens that entered the editor in insertions of more than tau tokens: with tau = 2, mean 20.6%, writers at or above 25/50/75%: 26.2%, 17.8% (95% CI 13.1–22.2), 11.6%; with tau = 5 (R28): 20.1%, 26.2%, 17.5%, 10.5%.

## F-23, F-24 and F-67 on the whole trace (src/r29_flora.py audits; src/r17_flora.py unchanged except for its output paths)

- F-23 new keystroke audit of T and X, drawn from all writers with keystrokes (seed 2171, at most three passages per writer, 50 per stratum; passages of the earlier samples excluded): token rule, T typed 93.8% (95% CI 86.7–98.8; 80 passages, 43 students), X not typed 85.7% (78.1–92.6; 119, 78); n-gram rule, T 70.1% (97, 47), X 78.4% (102, 77). With the first sheets (R17): 94.6% (37, 14) and 80.6% (124, 41).
- F-24 census of echo passages: writer-side evidence before the reply for 70.0% (62.1–77.3) of 938 passages from 130 students (first sheets: 64.5% of 279 from 38); prompt-first, large entries 64.0% (564 passages, 85 students); editor-first, large entries 68.0% (181, 43). Three echo runs with no covering 8-gram in a reply (editor route) or a prompt (prompt route) cannot be scored and were skipped; none occurred in the first sheets, where R17 ran without this guard.
- F-67 census of every T and X run of at least eight tokens: T typed 96.3% (91.5–99.6) of 323 runs from 87 students (98.4% of tokens); X not typed 93.8% (91.5–95.9) of 1,415 runs from 230 students (96.4% of tokens).

## F-69 pastes by dominant class (src/r29_flora.py pastes; src/r11_flora.py unchanged except for its output paths)

On the whole trace: 462 C-dominated pastes (90 students), 638 X-dominated (161), 91 mixed (47), 595 other (129). Medians: length 58 against 66 tokens; hours before the last snapshot 8.4 against 1.3; minutes after the latest reply 5.6 against 54.2. The labels written by the patched script are identical to the released labels.
