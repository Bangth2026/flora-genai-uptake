# Analysis log entry R15

Written after all earlier results (S1 to R14) were known and after its analyses had run, in response to an internal review of the manuscript before submission. All analyses are exploratory. No measure, class definition, parameter value or code of the attribution changes; every earlier result is unchanged. The benchmark script rebuilds the simulated students of seeds 2100 and 2200 exactly and reproduces the AURC values of R11 and R12.

## B. Benchmark (src/r15_benchmark.py)

B-17. Routes on which the definition of a class and the condition that assigns it diverge. In R10 to R14, typed text is generated in small edits and the rule for T is an entry size of at most tau, and other outside text is generated as a large unmatched paste, which is the condition for X; accuracy on these routes verifies the code, it does not validate the definitions. Three routes are added at a weight of 0.05 each to the 13 specified routes (seed 2300, 400 students): outside text retyped token by token (true X), pasted in pieces of 2 to 5 tokens (true X), and pasted in one insertion with alternating chunks of 8 to 30 tokens of a chatbot reply (true X for the outside chunks, C for the reply chunks).

Result: retyped and chunked outside text was T on 100% of its tokens; mixed outside chunks stayed X (99.8%), reply chunks C (96.7%). T precision 0.469 (0.997 without these routes); C precision 0.998, recall 0.864.

B-18. Routes that violate assumption A3 (seed 2400, 400 students, 13 specified routes plus three at 0.05 each, all true X): a passage identical to a reply of the embedded chatbot, pasted from an outside source 5 to 300 s after the reply (A3-after) or 2 to 60 s before it (A3-before), and a version of the reply with 15% of tokens replaced, pasted after it (A3-regen).

Result: A3-after 98.8% C; A3-before 65.8% E and 34.1% C (the C part had occurred in an earlier logged reply of the same student, since C requires a reply no later than entry); A3-regen 51.2% C, 48.6% X. C precision 0.685, recall 0.856; student-level U MAE 0.112.

B-19. Selective risk in the standard form (El-Yaniv & Wiener, 2010): risk over decided tokens only, curve ending at the method's coverage; area normalised by the coverage range, over the method's own range and over the common range [0, c*], c* the smallest coverage among the attribution and earliest plausible attribution with abstention by entry size. The convention of R11 and R12 (abstentions decided last and scored as "not chatbot text") is reported alongside.

| Set | Convention AURC: attribution / EPA / EPA + abstention | Standard: common coverage; risk at it (attribution / EPA + abstention / EPA) |
|---|---|---|
| 13 routes (seed 2100) | 0.021 / 0.058 / 0.023 | 49.3%; 0.96% / 1.00% / 5.79% |
| 20 routes (seed 2200) | 0.065 / 0.116 / 0.070 | 53.7%; 4.96% / 5.07% / 11.93% |
| Outside text retyped, chunked, mixed (seed 2300) | 0.015 / 0.047 / 0.017 | 50.4%; 0.47% / 0.55% / 4.82% |
| A3 routes (seed 2400) | 0.177 / 0.117 / 0.168 | 56.0%; 19.97% / 20.61% / 11.49% |

B-20. Tolerant matching inside the decision list: first reply and turn times computed over 8-grams that match exactly or with one token replaced (one wildcard position). Decision "chatbot text", precision / recall / F1: 13 routes 0.910 / 0.937 / 0.923 (attribution 0.997 / 0.847 / 0.916); 20 routes 0.921 / 0.864 / 0.891 (0.999 / 0.750 / 0.857); outside-text set 0.881 / 0.926 / 0.903; A3 set 0.615 / 0.936 / 0.742. Marked as chatbot text by route (13 routes): paraphrase 10% 0.95 (attribution 0.69), 25% 0.61 (0.22), prompt-first echo 0.24 (0.01); 20 routes: in-place edit 20% 0.87 (0.46), 50% 0.18 (0.02).

B-21. Retrieval of reply windows by sentence embeddings, after Krishna et al. (2023): windows of 20 tokens, stride 10, all-MiniLM-L6-v2, cosine threshold 0.75 or 0.85 (not tuned); direction-aware variant discards the best reply window when a turn window sent before that reply matches the submitted window at least as well. Precision / recall / F1, 13 routes: 0.723 / 0.886 / 0.796 (0.75), 0.780 / 0.875 / 0.825 (0.75, direction-aware), 0.772 / 0.732 / 0.751 (0.85), 0.830 / 0.723 / 0.773 (0.85, direction-aware); 20 routes: 0.769 / 0.825 / 0.796, 0.817 / 0.815 / 0.816, 0.809 / 0.640 / 0.715, 0.860 / 0.632 / 0.729. Prompt-first echo marked as chatbot text at 0.75: 0.86 without and 0.46 with the direction check.

## F. FLoRA (src/r15_flora.py)

F-16. Stepwise decomposition of the share of students with at least half the proposal in the chatbot class, paired bootstrap over students (2,000 resamples, seed 2150). Every level reproduces earlier results exactly (naive 36.0%, U 20.0%; maximum absolute difference from `results/U_naive.npy` and `results/U_R1.npy` 0).

| Step | Share at or above half | Difference (pp) [95% CI] | Students crossing |
|---|---|---|---|
| Naive overlap | 36.0% | | |
| + time-aware echo (R1) | 24.4% | 11.6 [8.0, 15.3] | 32 |
| + shared source (R3), no order check | 24.4% | 0.0 | 0 |
| + order check (R2, R4) = U | 20.0% | 4.4 [2.2, 6.9] | 12 |
| All rules | | 16.0 [12.0, 20.4] | 44 |

The shared-source rule moves no student across the threshold whether applied before or after the order check. Median U: 21.3% (naive), 8.8% (after R1), 4.8% (U).

F-17. Distribution of U: mean 23.6%, quartiles 0, 4.8% and 40.6%, U = 0 for 29.1% of students, 9th decile 85.4%.

F-18. Replies of the embedded model to near-identical prompts of different students (prompts of at least 8 tokens, replies of at least 40 tokens; 2,722 prompt-reply pairs): 8 pairs with identical prompts and 10 with 8-gram Jaccard similarity of at least 0.8 (18 pairs, 12 students). Coverage of one reply by the other's 8-grams (larger of the two directions): mean 0.46% [0.0, 1.0], median 0, 90th percentile 2.2%, 3 of 18 pairs above 0, none at or above 25%. Random pairs of replies to different students (4,969): mean 0.03%, 1.4% above 0.

Check of the two counts of 118 in the manuscript: `results/r11_flora.json` F3_shared (S 8-gram occurrences that appeared in an earlier reply, against 34,053 C) and `results/r13_flora.json` F14 (8-grams of first-snapshot X text in one to four other proposals, of 33,486). Different quantities with the same value.

Reproduce: `python -I src/r15_flora.py DATA && python -I src/r15_benchmark.py DATA R8_TOKEN_CLASSES` (R8_TOKEN_CLASSES: `git show af6bdb9:results/flora_token_classes.csv.gz > FILE`; the benchmark needs `sentence-transformers` and downloads all-MiniLM-L6-v2; 17 min on two CPU cores).
