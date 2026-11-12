# Plan R7: analyses requested in the second round of peer review

Committed and pushed before any R7 analysis was run. All R1–R6 results are known. Both analyses are descriptive and exploratory; neither changes a measure, class definition or parameter value.

## H1. FLoRA: students with and without logged pastes (reviewer 2, comment 2)

Validation checks V1, V2, V4 and V7 rest on the 68 students whose paste events were logged. Compare them with the 207 students without logged pastes on: number of large insertions (transitions inserting more than 20 tokens), proposal length (tokens), chatbot words received, number of student turns, grade, English as first language, U\*, and the shares of classes C, E, T and X. Report medians (means for shares), the difference with a 95% bootstrap interval (2,000 resamples of students), the Mann–Whitney p value (chi-squared for first language), and a standardised difference (difference in means divided by the pooled SD). The assumption that the two groups paste alike is examined descriptively; no decision rule.

As a check on whether the paste-based validation depends on who was logged, also report the share of large insertions (> 20 tokens) that carry chatbot-first 8-grams, by group: this quantity does not need paste logs.

## H2. RelianceScope: agreement by response-use code (reviewer 2, comment 3)

For each human response-use code with at least eight segments: number of segments, share of segments with any segment uptake (Algorithm 4), mean segment uptake, and the one-against-rest AUC of segment uptake, each with 95% intervals from 2,000 bootstrap resamples of students.
