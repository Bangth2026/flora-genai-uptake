# Analysis log entry R19

Written after all earlier results (S1 to R18) were known, in response to a second review of the manuscript (21 November 2026). All analyses are exploratory. No measure, class definition, parameter value or code of the attribution changes, and every earlier result is unchanged.

## F. FLoRA (src/r19_flora_sub.py, src/r19_flora.py)

F-40. Share of students at or above 25%, 50% and 75% of the proposal under U and five alternative counts: echo before chatbot-first in the token priority; U plus large-entry echo (tau = 5); U plus all echo; U plus X-with-C (subtypes of R16 without the embedding flags, computed by src/r19_flora_sub.py, which executes src/r16_flora.py up to its F-20 block); U plus large-entry echo plus X-with-C. Student-bootstrap intervals (2,000 resamples, seed 2191).

F-41. The entry threshold tau at 3, 5, 10, 20: mean shares of T and X, large-entry echo, and the share at or above half of U plus large-entry echo.

F-42. Size of the insertion that brought each final token into the editor (1, 2-5, 6-20, > 20 tokens, not in the last snapshot), for all tokens, T or X tokens and X tokens; for X tokens of the students with editor keystrokes, the share whose insertion had keystroke support (printable keystrokes from max(previous snapshot, snapshot - 3 s) to snapshot + 1 s at least half of the inserted characters).

F-43. Completeness of the editor log (the six measures of R10 P-2) for students with and without logged pastes: medians, difference in medians and standardised mean difference with bootstrap intervals, replacing the p-values of R10.

## B. Benchmark (src/r19_writer_corpus.py, src/r19_benchmark.py)

B-39. The four test sets of R15 rebuilt with the same seeds, with every route that uses the writer's own text (T, the echo routes, interleaving, undo) drawing it from 1,156 arXiv abstracts of any category (Hugging Face gfissore/arxiv-abstracts-2021, 12 pages of 100 rows at offsets drawn with seed 2193; at least 40 tokens) instead of the typed runs of other FLoRA proposals. Per-class precision, recall and F1, student-level error of U, standard-form selective risk at the common coverage.

B-40. Decision cost of the decision "chatbot text" on the eight sets (original and independent writer text): per method, false chatbot labels and misses per student; expected cost per 1,000 tokens for r = 1, 2, 5, 10, 20, 50 (a false label costs r misses); r*, the ratio above which the attribution costs less than a method that makes more false chatbot labels, with a student-bootstrap interval (2,000 resamples, seed 2194). Methods: attribution, audit operating point, earliest plausible attribution, naive overlap, paste-log rule, tolerant matching, the R17 tuned classifiers trained on 13 and on 26 routes (not retrained). Abstention counts as "not chatbot text".

Deviation recorded before the results were used: the first full run returned r* = 0 for methods with fewer false chatbot labels than the attribution, where r* is not defined. src/r19_fix_rstar.py sets these entries to null; the definition is corrected in src/r19_benchmark.py. No other value changed.

## Figures (src/r19_make_figs.py)

Fig. 4 of the article (distribution of the share of chatbot text under naive overlap, removal of all prompt 8-grams, U, U plus large-entry echo, U plus X-with-C) and Fig. 5 (class composition and X subtypes).

Reproduce: `EMB_CACHE=FILE python -I src/r19_flora_sub.py DATA R8_TOKEN_CLASSES SUB.pkl && python -I src/r19_flora.py DATA SUB.pkl && python -I src/r19_writer_corpus.py WRITER.json && EMB_CACHE=FILE python -I src/r19_benchmark.py DATA R8_TOKEN_CLASSES WRITER.json WORK && python -I src/r19_fix_rstar.py && python -I src/r19_make_figs.py DATA OUT`. src/r19_benchmark.py was run once with QUICK=1 before the full run.

## Results

- F-40, share at or above half: U 20.0%; echo first 16.0%; + large-entry echo 33.5%; + all echo 36.0%; + X-with-C 34.5%; + both 47.3%.
- F-41: U plus large-entry echo at or above half 33.8% (tau 3), 33.5% (tau 5), 32.7% (tau 10), 31.3% (tau 20).
- F-42: final tokens by insertion size: 1 12.6%, 2-5 4.2%, 6-20 6.6%, >20 75.5%, not in last snapshot 1.1%; X tokens with keystroke support 0.02%.
- F-43: standardised mean differences between -0.17 and 0.13; all intervals within -0.43 and 0.41.
- B-39: C precision with independent writer text 0.998, 0.999, 0.998, 0.734 (13 routes, 20 routes, broken definitions, A3); T precision on broken definitions 0.621 (original writer text 0.469).
- B-40: r* against the classifier trained on 13 routes: 19.5 [10.4, 58.5] (13 routes), 23.9 [13.7, 53.6] (20 routes), 0.9 [0.7, 1.2] (broken definitions), 0.8 [0.6, 1.1] (A3), 16.4 [9.1, 33.4] and 14.2 [7.9, 33.3] (independent writer text).
