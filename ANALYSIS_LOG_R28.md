# Analysis log entry R28

Written on 25 November 2026, after the analyses below ran, for the fourteenth version of the manuscript, in response to a review of the thirteenth. Nothing in the attribution, its rules or its parameters changed.

## A reading error in every earlier entry

The FLoRA system-log trace (Figshare files 5.2) is released as two workbooks, `trace1.xlsx` and `trace2.xlsx`, each with four sheets (`Result 1` to `Result 4`; 7,542,124 rows in all). Every earlier entry read it as `DATA/trace.pkl`, which held only the first sheet of each workbook: 2,071,788 rows, 71 students, two of them with part of their events in other sheets. Earlier entries therefore saw the editor keystrokes, chatbot-panel selections and logged pastes of 71 students and treated the other 204 as students whose keystrokes or pastes were not logged. They were logged: with all eight sheets, 272 students have editor keystrokes, 259 have panel selections and 270 have logged pastes (2,795 paste events). `src/r28_read_trace.py` reads all eight sheets into `DATA/trace_full.pkl` (`results/r28_trace_reading.json`); `DATA/trace.pkl` is left as it was, so earlier entries can be rerun.

The trace is used only for process evidence. The labels C, E, S, T and X, every class share, Table 2 to Table 6 of the article and the CoAuthor results do not read it and do not change.

## F. Process evidence repeated on the whole trace (src/r28_process.py; DATA_FULL holds the full trace as trace.pkl and the paste rows as pastes.pkl)

- F-62 (`src/r10_flora_checks.py` unchanged): for the 270 students with logged pastes, 94.9% of C tokens lay in logged pasted text (95% CI 91.5–97.6; 191 students with C tokens), 95.5% of X tokens (92.4–98.3) and 45.9% of T tokens. With the first sheets: 97.4%, 94.1% and 56.2% for 68 students.
- F-63 (`src/r12_flora.py` unchanged): 462 C-dominated pastes (90 students) and 638 X-dominated (161 students). X-dominated pastes came a median 1.3 h before the last snapshot against 8.4 h (mixed model ratio 0.31, 95% CI 0.24–0.40) and 3,255 s after the latest reply against 333 s (ratio 5.7, 4.2–7.8); with the first sheets 2.3 h against 17.9 h and 1,775 s against 326 s.
- F-64 (`src/r23_flora.py`, F-52 to F-54, unchanged): the data-driven entry threshold on 234 writers (297,485 insertions, 249,735 typed) is τ̂ = 2 (bootstrap 1–2; first sheets: 63 writers, τ̂ = 2, 1–4). Typed share by insertion size: 85.4% of one-token insertions, 29.3% of two-token, 1.2% of six to twenty, none above twenty. Census of chatbot-first runs of at least eight tokens: 1,280 runs from 193 students; selected in the chatbot panel before entry 82.4% (77.6–86.5; 94.2% of tokens), not typed 82.6%, typed 5.6%, paste shortcut 32.7%, carried by a logged paste 85.2% of 1,257 runs (first sheets: 354 runs from 57 students, selected 81.6%). The seven non-selected passages of the R9 audit are unchanged.
- F-65 trace coverage: 275 students in the trace; 272 with editor keystrokes; 270 with logged pastes; 259 with panel selections.
- F-66 (`src/r28_xsub_paste.py`): share of each subtype of X inside logged pasted text, 270 students: X 95.5%, X-log-gap 81.3%, X-reentered 95.2%, X-with-C 96.2%, X-cohort-reply 92.7%, X-first-snapshot 99.8%, X-no-evidence 93.3% (Table 5 of earlier versions: 93, 65, 87, 94, 98, 100 and 92 for 68 students).

The audit samples of R9 and R12 (50 C passages from 28 students; T and X samples; the echo census) were drawn from the 71 students of the first sheets and are not repeated.

## F. Counts under nested assumptions (src/r28_flora.py, run on DATA_FULL)

- F-57 subgroups: with the whole trace only 3 students lack keystrokes and 5 lack logged pastes, so the comparison of the audited subgroup with the rest no longer applies. Run on the first sheets (`python -I src/r28_flora.py DATA firstsheets`, `results/r28_flora_firstsheets.json`; 70 keystroke students against 205), it found no difference: mean C 27.3% against 22.4% (difference 5.0 points, 95% CI −3.9 to 15.1), share at or above half 22.9% against 19.0% (3.8, −7.6 to 15.5).
- F-58 the 55 writers at or above half chatbot-first text: all have keystrokes, 53 logged pastes; for 54 of the 55, at least 80% of the tokens of their chatbot-first runs had been selected in the panel before entry (median 100%).
- F-59 a lower end that leaves out C tokens that entered in insertions of at most τ tokens (words typed inside or at the edge of a chatbot-first passage; 14.2% of C tokens, 3.6% of the mean proposal): mean 20.1%; writers at or above 25/50/75%: 26.2%, 17.5% (13.1–22.2), 10.5%.
- F-60 τ = 2: mean T 10.3%, X 52.3%; C, E, S and X-with-C as at τ = 5; the share at or above half without B 82.5% (77.8–86.9) against 81.8%; other rows unchanged.
- F-61 token priority: C before E 20.0%; E before C 16.0% (R23).

## B-48 tolerant matching with a confidence score (src/r28_benchmark.py)

Students of seeds 2100 and 2200 rebuilt as in R16–R18 (R8_TOKEN_CLASSES: `git show af6bdb9:results/flora_token_classes.csv.gz`; B-29 reproduced exactly). Tolerant matching abstains on X and ranks a decided token by the number of covering 8-grams that carry its label under exact matching (0 if exact matching gives another label). Selective risk at the common coverage c* (0.455; 0.469): 0.85% on the 13 specified routes (attribution 0.85%), 4.64% with all 20 routes (attribution 5.05%). At the attribution's coverage (0.516; 0.555): 3.66% against 1.05%, and 6.40% against 4.95%. Coverage 0.555 and 0.607; F1 with every decided token 0.923 and 0.891.

## A. CoAuthor (src/r28_coauthor.py; readers, rules and parameters of R24 and R25)

- A-13 the lower end of F-59 in CoAuthor: mean 24.2% (C 24.4%); the logged share lay below it in 2.2% of sessions (below C: 7.1%); with it the narrow range held the logged share in 34.0% of sessions, without B 54.5%, without B and A1 61.4%.
- A-14 conservative intervals (half-widths times the square root of 1,445/63, the design effect if a writer's sessions were perfectly correlated): narrow range 29.1% [17.8, 40.4], without B 49.6% [37.0, 62.2], without B and A1 56.5% [44.3, 68.1].
- A-15 replies limited to the suggestion selected at each suggestion-select event (the suggestion at `currentHoverIndex` of the latest suggestion-open event, which matched the inserted text in 98% of 1,758 acceptances of the first 200 files; `currentSuggestionIndex` matched in 80%, and a first run that used it, with recall 71.8%, was discarded): C precision 98.9%, naive precision 98.3%, recall 84.9%, recall for suggestions of at least 8 tokens 93.7%, rank correlation 0.945 (naive 0.944), mean C 23.8%, X 3.1%; the narrow range held the logged share in 28.1% of sessions and the range without B in 49.6%. The main reader's values (A-15, all suggestions shown) are C precision 98.7%, naive precision 96.4%, recall 87.4%, recall for suggestions of at least 8 tokens 96.4%, rank correlation 0.968 (naive 0.961).
