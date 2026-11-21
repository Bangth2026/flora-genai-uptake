# Analysis log entry R20

Written in response to comments on the fifth version of the manuscript (21 November 2026), after test runs on small sets (QUICK=1) and before the full runs; committed with the results. All analyses are exploratory. No measure, class definition, parameter value or code of the attribution changes, and every earlier result is unchanged (the four test sets of R15 are rebuilt exactly; the attribution's risk at the common coverage reproduces 0.85% on the specified routes).

## B. Benchmark (src/r20_benchmark.py, src/r20b_benchmark.py)

B-41. The assumption-A3 set with each of the three A3 routes at weight 0.01, 0.05 and 0.10 (seed 2400): C precision, recall, F1, false chatbot labels and the share on A3 routes, share of each A3 route in each class.

B-42. Structural editing of chatbot text: five routes in which whole reply sentences (raw reply split at . ! ? : and line breaks) are edited before pasting: a 3-8 token span deleted from each sentence of at least ten tokens with probability 0.6; sentences shuffled; sentences of two replies alternated; the first 2-5 tokens of each sentence replaced by the writer's own words; all three. Each at weight 0.05 added to the 13 specified routes (seed 2800). Share of the surviving reply tokens in each class.

B-43. Route mixes far from FLoRA: a light-editing cohort (C-paste 0.40) and a typing cohort (T 0.45, M 0.08), seeds 2810 and 2820. Attribution, n-gram EPA and the R17 classifiers (not retrained): precision, recall, F1, student-level signed error of U, agreement on "at least half".

B-44. Standard-form selective risk at the attribution's own coverage for the methods that reach it (13 routes, 20 routes, broken definitions, A3).

B-45. Risk-coverage of the R17 classifiers ordered by max(p, 1 - p) for the decision "chatbot text" (p = predicted probability of C; chatbot text when p >= 0.5): risk and normalised area at the common coverage of R17 B-29, risk at the attribution's coverage, full-coverage error. src/r20b_benchmark.py runs B-44 and B-45 on the broken-definitions and A3 sets with the same code.

## F. FLoRA (src/r20_flora.py, src/r20_wikiwho.py)

F-44. Earliest plausible attribution on the field logs: n-gram EPA over the two-author revision sequence of Property 3; the same without prompts; EPA as implemented in WikiWho 1.0.3 (PyPI), fed every snapshot as a writer revision and every prompt and reply as a revision that appends the dialogue text to the current editor text, with WikiWho's Wikipedia spam and vandalism filters switched off, tokens aligned to the attribution's tokens. Mean, median, share of students at or above 25/50/75% (student bootstrap, 2,000 resamples, seed 2201), rank correlation with U, token agreement.

F-45. Echo split by whether the token entered the editor after a reply containing it (first editor time not earlier than first reply time): shares, and share of students at or above half when C is joined by that echo (all, large-entry) and by large-entry echo that did not.

F-46. C 8-grams with at least four numeric tokens or at least six function words (scikit-learn English stop words): share of C 8-grams and of C tokens covered only by such 8-grams.

F-47. Coincidence with replies to similar prompts: each prompt matched to the most similar prompt of another student (TF-IDF, word 1-2-grams, cosine); coverage of the proposal by 8-grams of the matched replies, with and without shared sources, against the replies of a random other student (five draws).

Reproduce: `python -I src/r20_flora.py DATA && EMB_CACHE=FILE python -I src/r20_benchmark.py DATA R8_TOKEN_CLASSES WORK && EMB_CACHE=FILE python -I src/r20b_benchmark.py DATA R8_TOKEN_CLASSES WORK` (R8_TOKEN_CLASSES: results/flora_token_classes.csv.gz at commit af6bdb9, as in R15-R19).

## Results

- B-41: C precision 0.920, 0.685, 0.560 at weights 0.01, 0.05, 0.10 (A3 tokens 3.9%, 17.1%, 28.6%); 97.6-99.7% of false chatbot labels on A3 routes.
- B-42: share labelled C of surviving reply tokens: deletion 0.94, reorder 0.87, merge 0.88, new openings 0.82, all 0.62; random replacement of 10% / 25% in the same set 0.69 / 0.25; set C precision 0.999, recall 0.853.
- B-43: light-editing C 0.999 / 0.934, U error -3.8 points; typing C 0.998 / 0.871, U error -2.5; n-gram EPA identical to the attribution; classifier (13 routes) F1 0.993 and 0.986.
- B-44: risk at the attribution's coverage (0.516) on the specified routes: attribution 1.05%, EPA 5.64%, dialogue only 7.10%, direction-blind 7.54%, naive 11.34%.
- B-45: classifier trained on 13 routes, risk at c*: 0.01% (13 routes), 0.03% (20 routes), 0.13% (broken definitions), 15.99% (A3); attribution 0.85%, 5.05%, 0.33%, 18.60%.
- F-44: share at or above half: U 20.0%; n-gram EPA two-author 20.0% (token agreement 99.9%); EPA without prompts 23.6%; WikiWho 12.7% (rho with U 0.73; 83.5% of its chatbot tokens are C; it credits 55.9% of C tokens).
- F-45: echo entered after a reply containing it 3.9% of the mean proposal (32.5% of large-entry echo); C plus it: 23.6% at or above half; C plus large-entry echo that did not: 28.7%.
- F-46: 0.15% of C 8-grams numeric, 0.27% function words; 0.09% of C tokens covered only by such 8-grams.
- F-47: similar-prompt replies cover 0.23% of a proposal (0.14% outside shared sources), random other student 0.11% (0.06%).
