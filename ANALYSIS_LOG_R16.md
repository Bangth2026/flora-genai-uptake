# Analysis log entry R16

Written after all earlier results (S1 to R15) were known, in response to a second internal review that asked for (i) a more interpretable treatment of the unresolved class, (ii) operating points for audit and screening, (iii) a systematic sensitivity analysis, (iv) baselines tuned on data separate from the test data and reported with intervals, and (v) a comparison with a classifier that learns from the same evidence. The scripts were written and run once on 40 simulated students per set before the full analyses ran. All analyses are exploratory. No measure, class definition, parameter value or code of the attribution changes; the students of seeds 2100, 2200, 2300 and 2400 are rebuilt exactly as in R15, and every earlier result is unchanged.

New simulated sets (400 students each): seed 2500, 13 specified routes (training); seed 2600, all 26 routes (the 20 of R11 plus the six of R15; training); seed 2700, 13 specified routes (validation for thresholds). No parameter was chosen on a test set.

## Shared code (src/r16_features.py)

Per token of the final artefact, 25 evidence features: the attribution's class (one-hot); the number of covering n-grams with each n-gram label; the number of covering n-grams found in a reply, in a writer turn and in a reply given to another writer; log entry size and an indicator that the token is not in the last snapshot; entry with the first non-empty snapshot; entry in the same inserted segment as a chatbot-first token, and the log size of that segment; the tolerant-matching mark of R15 B-20; the embedding similarities of R15 B-21 to the writer's own reply windows (without and with the direction check) and to reply windows given to other writers; the signed log time between the first reply containing a covering n-gram and its first editor time; and whether a covering n-gram was in the editor before the snapshot that brought the token in.

Subtypes of X, a reporting partition by the first condition that holds: X-log-gap (not in the last snapshot), X-reentered (a covering n-gram was in the editor earlier: moves, undo/redo, deleted and restored text), X-with-C (entered in the same inserted segment as chatbot-first text), X-near-own-reply (direction-checked embedding similarity to an earlier reply of the same writer at least theta_own), X-cohort-reply (a covering 8-gram in a reply given to another writer, or embedding similarity to such a reply at least theta_cohort), X-first-snapshot (entered with the first non-empty snapshot), X-no-evidence.

## B. Benchmark (src/r16_benchmark.py)

B-22. Subtypes of X by route on the four R15 test sets. theta_own and theta_cohort tuned on seed 2700 for F1 of the flag against chatbot routes and against the outside-AI route among X tokens: theta_own = 0.70 (validation F1 0.55, precision 0.47), theta_cohort = 0.86 (F1 0.86). On the 13-route set the true class of X-with-C tokens is C for 92.6%, of X-near-own-reply for 11.1%, of X-cohort-reply for 0%; X-reentered is true T for 99.3%.

B-23. Operating points for the decision "chatbot text": audit (decide C only when all eight covering 8-grams carry C, and E), screening (every class except X), full (every token). 13 routes: coverage 29.5% / 51.6% / 100%, precision of C 1.000 / 0.997 / 0.997, recall 0.550 / 0.847 / 0.847, selective risk 1.33% / 1.05% / 5.39%.

B-24. Sensitivity of the rule base: n in {5, 6, 8, 10, 12} at tau = 5, delta = 0; tau in {3, 5, 10, 20} at n = 8; delta in {0, 5, 30} s at n = 8; and n in {5, 8, 12} crossed with tau in {3, 10, 20}. Shared sources recomputed for each n. C precision was at least 0.987 in every configuration on both sets; recall ranged from 0.801 (n = 12) to 0.885 (n = 5) on 13 routes and 0.694 to 0.807 on 20; tau changes only the T/X split (T precision 0.73 at tau = 20).

B-25. Embedding retrieval thresholds tuned on seed 2700 over 0.60 to 0.95 (0.75 chosen for both variants). Student-bootstrap intervals (2,000 resamples) for the attribution, tolerant matching and tuned retrieval, and paired intervals for the difference in F1.

B-26. Evidence fusion: HistGradientBoostingClassifier (300 iterations, learning rate 0.1, 31 leaves, random_state 2160) and multinomial logistic regression on standardised features, each trained on seed 2500 (13 routes) and on seed 2600 (26 routes), tested on the four R15 sets. Reported: per-class and chatbot-text precision, recall, F1 with intervals and paired differences, student-level error of the share, selective risk ranked by |P(C) - 0.5| at the rule base's coverage, and marks by route.

B-27. Calibration: expected calibration error (15 equal-width bins) of the top label and of P(C); for the rule base, precision of C by the number of agreeing covering 8-grams.

B-28. Permutation importance (drop in chatbot-text F1, 60,000 tokens of the 13-route test set, one permutation per feature) for the tree model trained on 13 routes.

## F. FLoRA (src/r16_flora.py)

F-19. The attribution recomputed from the raw Figshare files with the R16 code agrees with the released token labels on 99.94% of tokens.

F-20. Subtypes of X on the 275 proposals with the thresholds of B-22: mean share of the proposal with student-bootstrap intervals, pooled share of X, number of students, and share inside logged pastes (68 students). Each condition is also reported on its own (not exclusive), with the overlap of X-near-own-reply with first-snapshot text.

F-21. The two tree models of B-26 applied to FLoRA without retraining: class shares, share of students at or above half, rank correlation with U, cross-tabulation with the rule labels, share of fusion-C tokens in logged pastes.

Reproduce: `EMB_CACHE=FILE python -I src/r16_benchmark.py DATA R8_TOKEN_CLASSES && EMB_CACHE=FILE python -I src/r16_flora.py DATA R8_TOKEN_CLASSES` (R8_TOKEN_CLASSES: `git show af6bdb9:results/flora_token_classes.csv.gz > FILE`; needs sentence-transformers and scikit-learn; about 30 and 6 minutes on two CPU cores; EMB_CACHE is an optional cache of window embeddings). The trained tree models are saved in `results/r16_fusion_models.pkl`.
