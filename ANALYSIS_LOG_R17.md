# Analysis log entry R17

Written after all earlier results (S1 to R16) were known, in response to a third internal review (simulated IPM review of 20 November 2026). The scripts were written, run once on 40 simulated students per set (QUICK=1), and then run in full. All analyses are exploratory. No measure, class definition, parameter value or code of the attribution changes; the simulated students of seeds 2100 to 2700 are rebuilt exactly as in R16, and every earlier result is unchanged.

## F. FLoRA (src/r17_flora.py, src/r17_flora_xsub.py)

F-22. The echo class by route (writer turn first, R1; editor first, R2) and by entry size (at most tau tokens or more), and the share of students at or above half under U, under U with the large-entry echo added, and under U with all echo added (student-bootstrap intervals, 2,000 resamples).

F-23. A new keystroke audit of T and X. Runs of at least eight tokens labelled T or X under the token entry rule and under the n-gram entry rule, from the 70 keystroke students, excluding every passage of the R9 and R9b samples (rebuilt with their seed 2091); new seed 2171; up to three per student and 50 per class and rule. Criteria fixed in R9: T agrees if printable editor keystrokes in the entry interval number at least half of the characters, X if fewer than a fifth. Each passage is scored for the label each rule gives it (majority label within the passage). Sample written to results/r17_audit_sample.csv.

F-24. Keystroke support of every echo run of at least eight tokens of the 70 keystroke students (a census), by route and entry size, with the criteria of R9.

F-25. Where large-entry echo came from: whether it entered the editor after a reply containing it had appeared; for students with logged pastes, whether it lay in a logged paste and whether that paste also carried chatbot-first 8-grams of an earlier reply; whether it occurs in a reply given to another student.

F-26. The peer threshold of the shared-source rule as a count (3, 5, 10) and as a share of the cohort (1%, 2%, 5%): share of the mean proposal in 8-grams of C or S that recur in at least that many other proposals.

F-27. Chance coverage of each proposal by the replies of 20 randomly drawn other students for n = 5 to 12 (mean, median and share of students above 1%), with the coverage by the student's own replies.

F-28 (src/r17_flora_xsub.py). The unresolved runs of at least 15 tokens that carry no subtype evidence (X-no-evidence, R16 thresholds), written to a file outside the repository for B-33. F-29: the subtypes of X with the two embedding flags switched off (results/r17_flora_xsub.json).

## B. Benchmark (src/r17_benchmark.py, src/r17_llm_baseline.py)

B-29. Risk--coverage in the standard form (risk over decided tokens only) for every method with a graded score on the four R15 test sets: coverage, selective risk at the smallest coverage among the methods compared, and the area over that common range divided by it; curves for the sets with 13 and 20 routes.

B-30. The attribution against earliest plausible attribution with abstention by entry size, standard form: paired student-bootstrap interval (300 resamples) for the difference in the area on all routes and with the shared-source route left out, and the difference with each route left out.

B-31. The A3 set: false chatbot labels of the attribution and of earliest plausible attribution with and without abstention, and the composition of the decided set at the common coverage.

B-32. Hyperparameter search for the evidence-fusion trees: learning rate {0.05, 0.1, 0.2} x leaves {15, 31, 63} x iterations {300, 600}, trained on seed 2500 and selected by chatbot-text F1 on seed 2700; the selected setting refitted on seeds 2500 and 2600 and tested on the four test sets (results/r17_fusion_tuned.pkl).

B-33. The simulator with other outside text from a different source: (a) the X-no-evidence runs of F-28; (b) arXiv abstracts in statistics and computing fetched from the Hugging Face dataset gfissore/arxiv-abstracts-2021 (random pages, seed 2173; at least 40 tokens). The four test sets are rebuilt with the same seeds; only the text of the routes that draw other outside text changes. Reported: per-class precision, recall and F1 of the rules, standard-form selective risk of the attribution and of earliest plausible attribution with and without abstention, and the R16 fusion models (not retrained) on the sets with 13 routes and with outside text retyped, chunked or mixed.

B-34. The first 20 students of the sets with 13 routes, with 20 routes and with the A3 routes, written for the LLM baseline.

B-35 (src/r17_llm_baseline.py). LLM baseline with the prompt of R11 B-7: Qwen2.5-3B-Instruct (GGUF, Q4_K_M) run with llama.cpp on CPU, greedy decoding; the first students of each set as stated in the results file; the first three students of the first set rerun to check that answers repeat.

B-36. Subtypes of X with the two embedding flags switched off, on the four test sets: share of X and share truly C.

Reproduce: `python -I src/r17_flora.py DATA && EMB_CACHE=FILE python -I src/r17_flora_xsub.py DATA R8_TOKEN_CLASSES XNOEV.csv && EMB_CACHE=FILE python -I src/r17_benchmark.py DATA R8_TOKEN_CLASSES XNOEV.csv ARXIV.json WORK && MODEL_PATH=qwen2.5-3b-instruct-q4_k_m.gguf python -I src/r17_llm_baseline.py WORK && python -I src/r17_make_figs.py OUT` (needs llama-cpp-python, sentence-transformers, scikit-learn, statsmodels).

## Results

(see the end of this file, written after the runs)

Results (written after the full runs; values in results/r17_*.json):

- F-22: students at or above half 20.0% [15.3, 24.7] under U; 33.5% [27.6, 38.9] with large-entry echo added; 36.0% with all echo. Echo tokens: 70% prompt-first, 82% entered in insertions > 5 tokens.
- F-23: 161 new passages (45 students). Token rule: T typed 95% [86, 100] (37 passages), X not typed 81% [70, 90]; n-gram rule: T 57% [41, 71], X 80% [69, 91].
- F-24: 279 echo runs (38 students): writer-side evidence 65% [51, 77]; prompt-first large 66%, editor-first large 40%.
- F-25: large-entry echo entered after a reply containing it 32.5%; for students with logged pastes 98% in pasted text, 72% in a paste that also carried chatbot-first 8-grams; 3.4% in replies given to other students.
- F-26: peer-recurrent C or S 8-grams cover 0.37% (3 students = 1% of the cohort) to 0.07% (14 students = 5%) of the mean proposal.
- F-27: mean chance coverage 0.95% (n = 5), 0.11% (n = 8), 0.003% (n = 12).
- F-29: subtypes without embedding flags, share of X: log gap 1.3%, re-entered 10.9%, with-C 24.6%, cohort reply 1.4%, first snapshot 22.5%, no evidence 39.2%.
- F-30: tuned trees on FLoRA, mean chatbot share 43.3% (13 routes) and 27.0% (26 routes).
- B-29: common coverage 0.455 (13 routes): risk 0.85% (attribution), 0.89% (earliest plausible + abstention), 5.76% (without); 20 routes 5.05%, 5.68%, 11.92%.
- B-30: area difference attribution minus earliest plausible + abstention: 13 routes -0.20 pp [-0.59, 0.07]; 20 routes -0.83 [-1.39, -0.55], without the shared-source route -0.69 [-1.23, -0.41].
- B-31: A3 set false chatbot labels 20,001 (attribution) and 20,003 (earliest plausible with and without abstention).
- B-32: selected learning rate 0.05, 15 leaves, 300 iterations (validation F1 0.981); test F1 0.983 (13 routes).
- B-33: C precision 0.997-0.999 without A3 routes and 0.678-0.689 with them for all three outside-text sources.
- B-35: LLM F1 0.783, 0.833, 0.608 on 12 students of the 13-route, 20-route and A3 sets; attribution 0.921, 0.915, 0.827 on the same students; answers identical on rerun.
- B-36: on the 13-route set, X-with-C truly C 93%, X-no-evidence 3%.
