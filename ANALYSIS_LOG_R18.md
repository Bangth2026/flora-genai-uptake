# Analysis log entry R18

Written after all earlier results (S1 to R17) were known, in response to a review of the fourth version of the manuscript (21 November 2026). All analyses are exploratory. No measure, class definition, parameter value or code of the attribution changes; the simulated students of seeds 2100 and 2200 are rebuilt exactly as in R16 and R17, and every earlier result is unchanged.

## F. FLoRA (src/r18_flora.py)

F-31. When FLoRA logged a reply. (a) Latency, logged reply time minus logged ask time, against the number of tokens in the reply: quantiles, Spearman correlation and a median regression on replies with latency between 0 and 600 s. (b) For every reply, its 8-grams that occur in no earlier reply and in no earlier or current writer turn, and that first reached the editor (first snapshot containing them, deleted later or not) after the ask: the time of first appearance minus the logged reply time. If replies were streamed and logged when complete, copying during streaming would place such 8-grams before the logged time. Reported: the share before the logged time, the share of replies with any such 8-gram (student-bootstrap interval, 2,000 resamples), the same count for an equally long span after the logged time, and a histogram of the differences.

F-32. The attribution with clock tolerance delta = +5, +30, +60 and +120 s: median and mean U, mean E, share of students at or above half (bootstrap interval), Spearman correlation with U at delta = 0, and the number of students whose U changes.

F-33. Snapshot transitions that changed the tokens of the text, by the number of tokens inserted (1, 2-5, 6-20, more than 20; deletions only), over all students; for the students with editor keystrokes, the share of insertions in each band that were typed: printable editor keystrokes between max(previous snapshot, snapshot - 3 s) and snapshot + 1 s at least half of the inserted characters.

F-34. The shared-source rule with the peer criterion (8-gram in at least five other proposals) not applied to 8-grams that the student's own chatbot produced and that the student did not supply first; highlights unchanged. Median and mean U, share at or above half, Spearman correlation with U.

Deviation recorded before results were used. The first full run of F-33 counted keystrokes strictly between the previous snapshot and the snapshot. It gave 24% of one-token insertions typed, which is implausible for an editor that saves after each change and pointed to the ordering of keystroke and snapshot events in the log. F-33 was rerun with the window of the R9 audit stated above; F-31, F-32 and F-34 are identical in both runs.

## B. Benchmark (src/r18_benchmark.py)

B-37. The paired difference in the standard-form area under the risk-coverage curve, attribution minus earliest plausible attribution with abstention by entry size (R17 B-30), with 2,000 paired student resamples instead of 300 (new seed 2184), on the sets with 13 and with 20 routes, with all routes and with the shared-source route left out, and with each route left out.

B-38. Student-bootstrap intervals (2,000 resamples, seed 2185) for chatbot-text precision, recall and F1 of the evidence-fusion classifiers of R16 (default settings) and R17 (tuned settings), trained on 13 and on all 26 routes and not retrained, and for the paired F1 difference with the attribution, on the sets with 13 and with 20 routes.

## Figure (src/r18_make_fig_example.py)

A worked example for the article: the synthetic student of demo/ with each passage coloured by its class and the log events that witness the label. It reproduces no study result.

Reproduce: `python -I src/r18_flora.py DATA && EMB_CACHE=FILE python -I src/r18_benchmark.py DATA R8_TOKEN_CLASSES WORK && python -I src/r18_make_fig_example.py OUT.png` (R8_TOKEN_CLASSES: `git show af6bdb9:results/flora_token_classes.csv.gz > FILE`). The benchmark script was run once with QUICK=1 (40 students per set) before the full run.

## Results

(written after the full runs; values in results/r18_flora.json and results/r18_benchmark.json)

- F-31: logged latency median 13.4 s, median regression 3.41 s + 0.0201 s per token (Spearman 0.68, 3,535 replies): the time is recorded when generation ends. Of 67,454 reply 8-grams that first reached the editor after the ask, 47 (0.07%; 2 replies of 2 students) did so before the logged reply time (42 more than 60 s before, 5 between 5 and 10 s before) and none in the 5 s after it.
- F-32: share of students at or above half 20.0% for delta = 0, +5, +30, +60 and +120 s; mean U 23.6%, 23.6%, 23.7%, 23.9%, 24.1%; students changed 0, 0, 2, 8, 21.
- F-33: 307,677 transitions that changed the tokens; 94.6% inserted one token, 1.3% two to five, 0.2% six to twenty, 0.6% more than twenty, 3.2% only deleted. Typed share (63 keystroke students): 84% (1 token), 34% (2-5), 3.9% (6-20), 0% (> 20).
- F-34: mean U +0.07 points, median U and share at or above half unchanged (4.8%, 20.0%), Spearman 0.999.
- B-37: area difference, attribution minus earliest plausible attribution with abstention: 13 routes -0.20 pp [-0.52, 0.08]; 20 routes -0.83 [-1.35, -0.55]; without the shared-source route -0.69 [-1.20, -0.44]; without the streaming route -0.46.
- B-38: tuned classifier, chatbot-text F1 [95% CI]: trained on 13 routes 0.983 [0.978, 0.987] (13 routes), 0.949 [0.940, 0.957] (20 routes); trained on 26 routes 0.949 [0.938, 0.959] and 0.947 [0.938, 0.956]. Paired F1 difference, attribution minus classifier: -0.067 [-0.079, -0.055], -0.033 [-0.045, -0.021], -0.092 [-0.104, -0.081], -0.090 [-0.103, -0.079]. Selective risks at the common coverage reproduce R17 B-29 exactly.
