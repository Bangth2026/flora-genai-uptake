# Five-class provenance of chatbot text in student writing (FLoRA) and programming (RelianceScope)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24101339.svg)](https://doi.org/10.5281/zenodo.24101339)

All versions: concept DOI 10.5281/zenodo.24101339.

Code, a dated analysis log and statistics files for a study that attributes each token of students' submitted work to five provenance classes (chatbot-first, echo of the student's own text, shared source, typed, untraced insertion) from dialogue and editor logs, validates the classes against simulated logs with known routes, keystrokes, chatbot-panel selections, logged pastes, a chance baseline and human codes, and relates verbatim uptake to grades, a post-test, dialogue features and AI-text detectors. Data: FLoRA Open Data IPS (Li et al., 2026; Figshare [10.6084/m9.figshare.32321499](https://doi.org/10.6084/m9.figshare.32321499); CC BY 4.0; 275 postgraduate students) and RelianceScope (Jin et al., 2026; 79 students; used with the authors' permission, not bundled here). All analyses are exploratory.

## Order of work (see `git log`)

| Commit | Content |
|---|---|
| `ANALYSIS_LOG_S1.md` | Log entry for Study 1, committed before any association was computed |
| `results/s1_stats.json` | Primary run as specified (naive uptake U) |
| Deviation D1 | Echo correction: exclude text the student pasted into prompts (`src/run_s1_echo.py`) |
| `src/align_check.py`, `ANALYSIS_LOG_S1b.md` | Clock check and log entry for provenance validation, committed before the provenance run |
| `results/s1b_provenance.json` | Provenance from the writing log; decision rule selects U_prov as the primary label |
| `results/s1_stats_uprov.json` | RQ2–RQ4 with U_prov |
| `ANALYSIS_LOG_R1.md`, `src/r1_clock.py` | Log entry R1 and clock check, committed before the R1 analyses |
| `results/r1_stats.json` | R1 analyses (`src/run_r1.py`) |
| `ANALYSIS_LOG_R2.md` | Log entry R2 (open AI-text detectors, third-source threshold), committed before the R2 analyses; deviations R2-1 and R2-2 recorded |
| `results/r2_stats.json`, `results/r2_sens.json` | R2 analyses (`src/r2_label.py`, `src/r2_detect.py`, `src/r2_stats.py`); post hoc check `results/r2_posthoc.json` |
| `ANALYSIS_LOG_R3.md` | Log entry R3 (five provenance classes C/E/S/T/X; second dataset with human response-use codes and post-test; FLoRA typed/untraced reanalysis declared post hoc), committed before the R3 analyses |
| `results/r3_rs_stats.json`, `results/r3_flora_stats.json`, `results/r3_d3swap.json` | R3 analyses (`src/run_r3_rs.py`, `src/run_r3_flora.py`, `src/r3_d3swap.py`) |
| `ANALYSIS_LOG_R4.md` | Log entry R4 (edit-size entry rule; T/X validation against logged pastes; chance baseline and bounds for code; length-controlled human-code validation; censored post-test model), committed before the R4 analyses |
| `results/r4_rs_stats.json`, `results/r4_flora_stats.json`, `results/r4_flora_tx_timing.json` | R4 analyses (`src/r4_rs.py`, `src/r4_flora.py`, `src/r4_flora_tx_timing.py`) |
| `ANALYSIS_LOG_R5.md` | Log entry R5 (moves within the document, token projection, clock distribution, code-aware tokenisation, origin of E, timing baseline, robustness of grade and post-test models, detector thresholds, size-matched and semantic paraphrase check), committed before the R5 analyses |
| `results/r5_*.json` | R5 analyses (`src/r5_flora_attr.py`, `src/r5_clock.py`, `src/r5_flora_grade.py`, `src/r5_flora_para.py`, `src/r5_rs.py`; figure `src/make_fig_classes_r5.py`) |
| `ANALYSIS_LOG_R6.md` | Log entry R6 (degraded logs, parameter choices, subdivision of X, origin of logged pastes, precision of outcome estimates, demonstration code), committed before the R6 analyses |
| `results/r6_flora.json`, `results/r6_rs.json`, `demo/` | R6 analyses (`src/r6_flora.py`, `src/r6_rs.py`) and an end-to-end demonstration on synthetic logs |
| `ANALYSIS_LOG_R7.md` | Log entry R7 (students with and without logged pastes; agreement by response-use code), committed before the R7 analyses |
| `ANALYSIS_LOG_R8.md` | Log entry R8 (misclassification of uptake, equivalence tests, order of the E and S rules, retyped code, label-noise ceiling, comparison with the LLM classifiers released with RelianceScope, pre-activity covariates, Tobit standard errors, token-level export), committed before the R8 analyses; addendum K5 and the Tobit correction were committed before they ran |
| `results/r8_flora.json`, `results/r8_rs.json`, `results/r8_rs_llm.json`, `results/r8_rs_tobit.json` | R8 analyses (`src/r8_flora.py`, `src/r8_rs.py`, `src/r8_rs_llm.py`, `src/r8_rs_tobit.py`) |
| `results/flora_token_classes.csv.gz`, `results/flora_student_classes.csv` | Five-class label of every token of the 275 FLoRA proposals (user_id, token_index, token, class) and per-student class shares, including X_with_C (untraced text that entered with chatbot-first text). Derived from FLoRA Open Data IPS (CC BY 4.0) and released under the same licence |
| `ANALYSIS_LOG_R9.md`, `ANALYSIS_LOG_R10.md` | Log entries R9 and R10, each committed before its analyses (R10: planted-provenance benchmark, external baselines, online version, RelianceScope intermediate snapshots and parameter table, paste checks with intervals) |
| `results/r10_*.json` | R10 analyses (`src/r10_benchmark.py`, `src/r10_online.py`, `src/r10_baselines.py`, `src/r10_rs.py`, `src/r10_flora_checks.py`; `demo/baselines.py`, `demo/provenance_online.py`) |
| `ANALYSIS_LOG_R11.md` | Log entry R11 (properties of the rules, routes outside the specification, route mixes, seeds and intervals, earliest-plausible, paste-log and LLM baselines, entry rules, risk-coverage; FLoRA unresolved class, shared-source rule, clock direction, audit clustering; token entry rule made primary), committed before the R11 analyses |
| `results/r11_*.json`, `results/fig4_classes_r11.png`, `results/fig5_risk_coverage_r11.png` | R11 analyses (`src/r11_benchmark.py`, `src/r11_flora.py`, `src/r11_llm_baseline.py`; figures `src/make_fig_classes_r11.py`, `src/make_fig_rc_r11.py`). `results/flora_token_classes.csv.gz` and `results/flora_student_classes.csv` re-exported with the token entry rule (n-gram entry kept as a column) |
| `ANALYSIS_LOG_R12.md` | Log entry R12 (selective prediction with a stated confidence for every baseline and abstention by entry size attached to the baselines; online version with delta > 0; FLoRA unresolved text, working span, U\* distribution, naive overlap at n = 5 and 12, paste comparisons with students as clusters; RelianceScope token entry rule). Written after its analyses ran; no measure, class definition or parameter changed |
| `results/r12_*.json`, `results/fig3_risk_coverage_r12.png`, `results/fig5_classes_r12.png` | R12 analyses (`src/r12_benchmark.py`, `src/r12_flora.py`, `src/r12_rs.py`, `src/r12_audit_selection.py`; figures `src/make_fig_rc_r12.py`, `src/make_fig_classes_r12.py`); `demo/provenance_online.py` holds labels for delta seconds after entry |
| `ANALYSIS_LOG_R13.md` | Log entry R13 (decision list applied as written when delta > 0: offline R2 with delta, online R1 for S labels; sources of the remaining risk-coverage difference by route; FLoRA unresolved text in the first snapshot, panel selections for the token-rule audit sample, overlap of each class with English Wikipedia). Written after its analyses ran; no result with delta = 0 changed |
| `results/r13_*.json` | R13 analyses (`src/r13_benchmark.py`, `src/r13_flora.py`, `src/r13_wiki.py`); `demo/provenance.py` and `demo/provenance_online.py` updated (D-3, D-4) |
| `results/r7_flora.json`, `results/r7_rs.json` | R7 analyses (`src/r7_flora.py`, `src/r7_rs.py`); figures `src/make_flow_fig.py` (text-flow diagram) and `src/make_fig_classes_r7.py` (class figure with panel b from the R4 attribution) |

## Main results (log entry R1, primary measure U*)

U* is the share of the final proposal covered by 8-grams that are in a chatbot reply, were not supplied by the student in an earlier prompt (time-aware echo correction), do not occur in reading highlights or in at least five other proposals, and were in a reply no later than their first appearance in the editor. Clocks aligned (chat-box Enter keystrokes vs ask times: median difference 2.2 s, `results/r1_clock.json`).

- A direction-blind echo correction (v1.0) removed 25,994 proposal 8-grams, of which 9,533 (36.7%) had first appeared in a chatbot reply.
- Logged paste events (68 students) contain 97.1% of U*-covered tokens.
- Median U* 0.048; 30.5% of students take at least a quarter of the proposal from chatbot replies, 20.0% at least half; 69% of uptake tokens enter in the last 24 h before the last snapshot.
- Grade per SD of U*: 0.02 points on 0–15 (95% CI −0.21 to 0.24); mean grade by uptake quartile 11.9, 11.8, 11.8, 11.7.
- Detection of the top quartile: exposure AUC 0.615 (0.538–0.693); coded intent and request features add nothing detectable (all gains inconclusive).

## Five provenance classes (log entry R3)

Each token of the final artefact is labelled C (chatbot-first), E (echo of the student's own text), S (shared source), T (typed in the editor) or X (untraced insertion). U* is the share of class C.

| Mean share of tokens | C | E | S | T | X |
|---|---|---|---|---|---|
| FLoRA proposals (n = 275) | 23.6% | 13.6% | 0.2% | 12.8% | 49.8% |
| RelianceScope code (n = 79) | 23.4% | 5.0% | 42.9% | 28.6% | 0.1% |

RelianceScope validation: segment uptake vs human codes "copy / write over the response" AUC 0.759 [0.713, 0.799]; U* vs perceived reliance ρ = 0.65. Post-test on pre-test and U*: −0.32 per SD [−0.71, 0.06] (58% at ceiling). FLoRA detectors, chatbot vs typed paragraphs: AUC 0.805 (D1), 0.665 (D2), 0.710 (D3); untraced paragraphs score like chatbot paragraphs.

Log entry R4 (entry mode by edit size): FLoRA T 12.4%, X 50.2%; 89.7% of X 8-grams but 3.0% of T 8-grams entered through a logged paste. RelianceScope chance-corrected U* 13.2% (raw 23.4%); human-code AUC for "copy the response" 0.762, gain over a reply-length model 0.187 [0.139, 0.238]; post-test Tobit −0.75 per SD of U* [−1.37, −0.12].

Log entry R5 (checks of assumptions and robustness): moves within the document account for 1.1% of FLoRA X tokens (X 50.2% → 49.6%); a segment-local entry rule gives X 43.6%, T 19.0%. Token projection with E before C: median U* 2.8%, ≥ 50% 16.0% (ρ = 0.98 with U*). Clock: 99.0% of asks whose nearest Enter keystroke was > 30 s away had another chat-box keystroke within 30 s; no offset near 1 h. Grade: null under multiple imputation, missing indicators, Tobit at 15, rank/log/spline U*, and for X and T shares. RelianceScope post-test, per SD: raw U* OLS −0.32 [−0.71, 0.06], Tobit −0.75; U* excluding the reference solution OLS −0.59 [−1.01, −0.18], Tobit −1.04 [−1.58, −0.51]. Code-aware lexers: chance-corrected U* 8.5% (operators kept) and 1.8% (identifiers normalised); bound excluding the reference solution 11–16%.

The RelianceScope data (Jin et al., 2026; [10.1145/3774398.3811612](https://doi.org/10.1145/3774398.3811612)) are used with the permission of the dataset authors, who allow use and redistribution provided the paper is cited. They are not bundled here: obtain them from the authors' Open Science Framework repository and place them in RS_DIR. This repository holds only code and aggregate RelianceScope statistics; `src/run_r3_rs.py` also writes a student-level file (`results/r3_rs_students.csv`), which is excluded from version control and must not be shared.

In v1.8.0 the headings of log entries R6 and R7 were reworded to describe their content; their analysis specifications are unchanged, and the original files are in the git history at the commits listed in Table S3 of the article's Supplementary Material.

## Degraded logs and parameter choices (log entry R6)

No measure or class definition changed. `src/r6_flora.py` reproduces U\* for every FLoRA student exactly.

- Log resolution: with snapshots thinned to ≥ 5, 15, 60, 300 s apart, U\* keeps its ranking (ρ = 0.999, 0.999, 0.995, 0.974); the T/X split agrees with the full-resolution split at κ = 0.98, 0.84, 0.47, 0.12 (τ = 5) and 0.92, 0.92, 0.80, 0.35 (τ = 20).
- Dialogue only (no snapshots): median 8.8% vs 4.8% for U\* (ρ = 0.94); 16.3% of its chatbot tokens are not chatbot-first; 31.7% of echo tokens become chatbot text. RelianceScope (code attached to each message): equal to U\* (ρ = 1.00).
- Parameters: grade slope per SD between −0.02 and 0.08 under every n, δ, peer threshold, projection, entry rule and τ (all intervals include zero); X share 51.3% (τ = 1) to 46.3% (τ = 50); students above own null 99th percentile 70.9%, 65.8%, 56.4% at n = 5, 8, 12.
- Untraced class: 37.6% of X 8-grams entered in the same paste as chatbot-first text; 71.5% in segments > 50 tokens.
- Logged pastes (≥ 8 tokens, n = 687): 30.6% mostly chatbot text, 19.8% moved within the document, 39.3% from outside the logs, 10.3% mixed; 73.1% of C tokens entered through a paste that was mostly chatbot text.
- Precision: FLoRA grade 0.005 points per 10 percentage points of uptake (CI −0.06 to 0.07), minimum detectable slope 0.32 points per SD; RelianceScope minimum detectable slope 0.55 (OLS) and 0.89 (Tobit) points per SD; simulated Tobit power 33% for −0.50 and 64% for −0.75 per SD.

## Paste-logged students and agreement by code (log entry R7)

- Students with (68) and without (207) logged pastes: no difference in grade (mean 11.81 both), proposal length (median 715 vs 703 tokens), large insertions (median 4 vs 3), untraced share (46.4% vs 51.4%) or English as first language (14.9% vs 9.5%, p = 0.32). Logged students received more chatbot text (median 5,515 vs 4,714 words, SMD 0.30, p = 0.04) and had higher median U\* (11.8% vs 3.4%, p = 0.06). Share of large insertions carrying chatbot-first text: 26.5% vs 24.3%.
- RelianceScope agreement by response-use code (one against rest, segment uptake): copy the response AUC 0.76 [0.72, 0.80]; write over the response 0.55 [0.49, 0.61]; correct artefact 0.39; create artefacts by oneself 0.32; read and move on 0.31. Share of segments with any uptake: 73%, 47%, 19%, 6%, 3% (edit-size entry rule; the R5 figure used the R3 rule).

## Misclassification, retyped code and LLM classifiers (log entry R8)

- FLoRA grade (OLS, HC3, per SD): C + 0.25X 0.04 [−0.19, 0.26]; C + 0.5X 0.06 [−0.16, 0.29]; C + X 0.08 [−0.16, 0.32]; C + X_C −0.01 [−0.25, 0.24]; C and X_C together 0.01 and −0.02. X_C (untraced text entering with chatbot-first text) is 36.0% of X tokens, 17.4% of the mean proposal. TOST against ±0.25 grade SD: p < 0.001 for U\* and every composite; smallest equivalence bound 0.10 grade SD for U\*. Tobit at 15 with numerical-Hessian SE: 0.02 [−0.21, 0.26].
- Order of the E and S rules: no n-gram in either dataset was both in a shared source and in the editor before the first reply containing it, so the order changes nothing.
- RelianceScope, write over the response (92 segments): AUC 0.55 at n = 8, 0.57 at n = 5, 0.62 with the code lexer at n = 5; widening the segment window or counting reply n-grams in the final code regardless of class does not help. Of 49 write-over segments without uptake, 48 had code blocks, 3 had any reply 8-gram in the final code, 16 a reply 5-gram entering the editor during the segment: retyped code was mostly replaced before submission.
- Label-noise ceiling for the copy code (simulated AUC of a perfect score): 0.93, 0.87, 0.81, 0.79 at label error 5%, 10%, 15%, 17%.
- Released LLM classifiers (145 matched benchmark segments, passive response use): segment uptake > 0 precision 0.88, recall 0.50, F1 0.64; best LLM configuration F1 0.77, median 0.62.
- Post-test with pre-activity questionnaire (self-efficacy, motivation, metacognition) instead of task progress, per SD: raw U\* OLS −0.23 [−0.68, 0.22], Tobit −0.56 [−1.19, 0.07]; U\* excluding the reference solution OLS −0.50 [−0.99, −0.02], Tobit −0.85 [−1.39, −0.31].
- Tobit standard errors: the BFGS inverse-Hessian approximation used in log entries R4–R6 understated the uncertainty of models with several covariates; R8 reports numerical-Hessian standard errors (`results/r8_rs_tobit.json`). With the pre-test alone, intervals move by at most 0.04 points.

## Analysis log, not pre-registration (v1.9.0)

The files `ANALYSIS_LOG_*.md` were named `PREREG_*.md` up to v1.8.0 and were renamed; file names cited inside them were updated, and their content is otherwise unchanged. Each entry was written as the analysis proceeded, after the results of earlier entries were known, and first results usually followed within minutes; commit times show the order of commits, not that an analysis had not been tried locally before. The entries are a record of decisions, not pre-registrations, and all analyses are exploratory.

## Validation against known routes and independent process evidence (log entry R9)

- Simulated logs with known routes (`src/r9_synthetic.py`, `results/r9_synthetic.json`; 300 students per condition): with complete logs on one clock, C, E and S precision and recall ≥ 0.998; X recall 0.91 (boundary tokens become T); a dialogue clock 60 s ahead of the editor turns about a third of chatbot-first tokens into E; thinning removes the T/X split; replacing every 8th token of pasted chatbot text removes it from C.
- Keystroke and selection audit (`src/r9_flora.py`, `results/r9_audit_sample.csv`; 70 FLoRA students; 50 passages per class): C not typed 0.86 [0.76, 0.94] and selected in the chatbot panel before entry 0.86; E student side first 0.64 (prompt-first 0.73, editor-first 0.30); X not typed 0.94; T typed 0.48 per n-gram entry and 0.96 with token-level entry (`src/r9b_flora_token_entry.py`, added after the first audit result).
- Echo by entry mode: E-X 11.0% and E-T 2.6% of the mean proposal; students with at least half the proposal in C 20.0%, in C + E-X 32.7%, in C + E 36.0%.
- Run time of Algorithms 1–2: 51 s for 275 FLoRA students and 496,733 snapshots on one CPU core (0.1 ms per snapshot); 0.4 s for RelianceScope.
- Chance decomposition for code (`src/r9_rs.py`): convergence explains 43% (alphanumeric), 71% (code lexer) and 94% (identifiers normalised) of chatbot-first code.

## Planted provenance, external baselines and online operation (log entry R10)

No measure, class definition or parameter value changed.

- Benchmark with planted provenance fitted to FLoRA (`src/r10_benchmark.py`, `results/r10_benchmark.json`; 400 simulated students built from real FLoRA replies, proposals, highlights and editor timing; 13 routes): token accuracy 0.93; precision/recall C 0.997/0.847, E 0.945/0.994, S 0.923/0.971, T 0.879/0.989, X 0.892/0.959. Retyped chatbot text is recognised as C (0.97); paraphrase removes it from C (C 0.69 with 10% and 0.22 with 25% of tokens replaced, the rest falls in X). Student-level U* against the true C share: mean absolute error 0.051, bias -0.050, Spearman 0.92, agreement on "at least half" 0.92. With snapshots 60-300 s apart, editor-first echo that the environment passes to the chatbot becomes C (80-89%), and T recall falls below 0.10.
- Baselines on the same logs, chatbot text at token level (precision/recall/F1): attribution 0.997/0.847/0.916; dialogue only 0.924/0.848/0.885; direction-blind removal 0.923/0.839/0.879; naive 8-gram overlap 0.746/0.860/0.799; string tiling (MML 8) 0.748/0.860/0.800; winnowing (k 8, w 4) 0.749/0.796/0.772. Typed share: attribution MAE 0.019; 1 minus the overlap share MAE 0.48.
- Study data (`src/r10_baselines.py`, `src/r10_rs.py`): in FLoRA, winnowing and string tiling mark a median of 16.0% and 20.6% of the proposal (U* 4.8%); 32.4% and 36.0% of students at or above half (U* 20.0%); 37% to 38% of the marked tokens are echo. In RelianceScope, segment agreement with "copy the response": segment uptake AUC 0.762 [0.724, 0.805], string tiling 0.693 [0.646, 0.740], winnowing 0.674 [0.627, 0.723]; chance level 8-9% of the final code.
- Online version (`demo/provenance_online.py`, `src/r10_online.py`): labels agree with the primary offline labels on 99.3% of FLoRA tokens (kappa 0.99; U* Spearman 0.998) and on all RelianceScope tokens; 1.9% of n-gram labels change after entry (editor-first echo). Latency per snapshot: median 0.06 ms, 99th percentile 0.23 ms, maximum 9 ms (FLoRA, one CPU core); peak memory per student median 3.2 MB (max 13 MB) online and 1.1 MB offline; 40 s online and 33 s offline for the 275 FLoRA students and 496,733 snapshots.
- RelianceScope (`src/r10_rs.py`): labelling the snapshot at the end of each segment instead of the final code leaves "write over the response" at AUC 0.55 (copy 0.78). Parameter table: chance-corrected mean uptake 13.2% (primary), 10.7% (n = 5), 11.4% (n = 12), 9.9% (E before C), 2.1% (majority), 8.5% (code lexer), 1.8% (identifiers normalised); tau has no effect on C; post-test slope per SD between -0.31 and -0.41, all intervals but E-before-C include zero. Label-noise ceiling for the copy code: 0.97, 0.94, 0.92, 0.89, 0.87, 0.85, 0.82, 0.80, 0.78, 0.76 at q = 0.02 to 0.20.
- Rule conflicts (`src/r10_rule_conflicts.py`): 23 of 16,461 prompt-first FLoRA 8-grams (0.14%) and 4 of 21 in RelianceScope also occur in a shared source; no n-gram is both shared and in the editor before its first reply.
- Figures 1, 2 and 4 relabelled: four evidential classes (C, E, S, T) and the unresolved outcome X (formerly "untraced").
- FLoRA paste checks with student-level intervals (`src/r10_flora_checks.py`, 68 students): C tokens covered by logged pastes 97.4% [94.6, 99.3]; X 94.1% [85.7, 99.3]; T 56.2% [40.2, 77.1]. Editor snapshots per 100 final tokens, interval between snapshots and share of large insertions do not differ between students with and without logged pastes (p > 0.13). Asks whose nearest Enter keystroke lay near 1, 10 or 11 h (6 asks, 4 students) all had another chat-box keystroke within 30 s.

## Properties, routes outside the specification and selective prediction (log entry R11)

The entry mode that separates typed (T) from unresolved (X) text is now decided per token (R9, F-8) instead of per n-gram; C, E, S and U* are unchanged. FLoRA class shares with the token entry rule: C 23.6%, E 13.6%, S 0.2%, T 11.7%, X 50.9% (n-gram entry: T 12.4%, X 50.2%; token agreement 93.6%, kappa 0.90).

- Properties (`src/r11_benchmark.py`): none of 17,487 n-grams that were in the editor before any reply contained them was labelled C (800 simulated students); every online label that changed after entry changed from S, T or X to E; two runs gave identical outputs.
- Seeds (20 x 400 students, R10 routes, token entry): C precision 0.997 (0.996-0.998), recall 0.845 (0.830-0.865), F1 0.915 (0.906-0.926); bootstrap F1 for seed 2100: 0.916 [0.904, 0.928].
- Routes outside the specification (7 routes, 400 students): C precision 0.999, recall 0.750; reply edited in place 20% / 50%: 46% / 2% of surviving reply tokens stay C, the rest X (T under the n-gram rule); regeneration with both versions logged 0.996 C, first only 0.50 C; copy while streaming with the reply logged at completion 66% E (delta = 0), 98% C with delta = 30 s; undo/redo of typed text becomes X under the token rule.
- Route mixes (50 Dirichlet mixes x 200 students): C precision >= 0.988 in every mix; recall 0.65-0.98 (13 routes) and 0.52-0.91 (20 routes), 5th-95th percentile.
- Baselines (F1, chatbot text, seed 2100): attribution 0.916; earliest plausible attribution 0.917 (difference -0.001 [-0.002, 0.000]); dialogue only 0.885 (+0.031 [0.025, 0.039]); naive overlap 0.799; string tiling 0.800; winnowing 0.772; paste-log rule 0.690. With the routes outside the specification, dialogue only 0.866 vs 0.857 for the attribution. LLM baseline: see `results/r11_llm_baseline.json`.
- Risk-coverage (AURC, lower is better): attribution 0.021 (error 1.1% on the 51.6% of tokens decided before X), earliest plausible 0.054, dialogue only 0.075, naive overlap 0.123; with routes outside the specification 0.065, 0.102, 0.116, 0.157.
- FLoRA keystroke audit with the token entry rule: T typed 0.96 [0.90, 1.00] (22 students), X not typed 0.94 [0.87, 1.00] (32 students); ICC of agreement within students from about 0 to 0.61.
- FLoRA pastes mostly X vs mostly C: median 71 vs 37 tokens, 2.3 h vs 17.9 h before the last snapshot, 29.6 vs 5.4 min after the last reply (all p < 0.01). X tokens entering in the last 24 h: 81% (C 69%, T 56%). X entering with chatbot-first text: 37.7% of X, 18.6% of the mean proposal.
- Shared-source rule: 118 S 8-gram occurrences had appeared in an earlier reply, against 34,053 C (0.35%). Clock: logged ask time follows the chat-box Enter keystroke by a median of 2.2 s (quartiles 1.5, 3.8); no C or editor-first E 8-gram lies within 5 s of its first reply, so delta in [-5, 5] s changes no label. P-1 on FLoRA holds by construction of the rules (delta = 0); the empirical count was run on the simulated students.

Reproduce: `python -I src/r11_benchmark.py DATA WORK && python -I src/r11_flora.py DATA && python -I src/r11_llm_baseline.py WORK 40` (local model; `LLM_BACKEND=api API_BASE=... API_KEY=... MODEL=...` for a hosted model).

## Demo without the study data

```bash
python demo/make_synthetic.py && python demo/run_demo.py   # or open demo/demo.ipynb
```
See `demo/README.md`. `demo/provenance.py` implements Algorithms 1, 2 and 4 with the standard library only.

## Open AI-text detectors (log entry R2)

Paragraphs of the final proposals (≥ 40 tokens) labelled from the logs: 147 chatbot (≥ 80% covered by chatbot-first 8-grams) and 618 composed (no reply 8-gram, < 5% logged paste).

| Detector | AUC [95% CI] | False flags at 50% / 80% detection |
|---|---|---|
| D1 mean log-likelihood, Qwen2.5-1.5B | 0.649 [0.586, 0.708] | 26.7% / 60.8% |
| D2 Fast-DetectGPT, Qwen2.5-1.5B | 0.533 [0.471, 0.596] | 46.8% / 75.2% |
| D3 Binoculars, Qwen2.5-0.5B / 0.5B-Instruct | 0.573 [0.501, 0.638] | 41.4% / 73.8% |

Composed paragraphs of students with a first language other than English scored as more machine-like on all three detectors (intervals exclude zero). Third-source threshold k = 3, 5, 10 leaves U* and the grade slope unchanged.

All numbers: `results/r1_stats.json`, `results/r2_stats.json`. Earlier versions of the measure: `results/s1_*.json`, `results/s1b_provenance.json`.

## Selective prediction, online operation and unresolved text (log entry R12)

No measure, class definition or parameter value changed. The rules are listed in the order the code has always applied them: R1 echo (turn first), R2 echo (editor first), R3 shared source, R4 chatbot-first, R5 no evidence.

- Risk-coverage with a stated confidence for every method (`results/r12_benchmark.json`, same simulated students as R11). 13 routes, AURC / error when every token is decided: attribution 0.021 / 0.054; earliest plausible attribution 0.058 / 0.053, with abstention by entry size 0.023 / 0.053 (difference from the attribution -0.001 [-0.004, 0.000]); dialogue only 0.075 / 0.121 (0.092 with abstention); direction-blind 0.079 / 0.124; naive overlap 0.123 / 0.179 (0.188 with abstention); paste-log 0.234 / 0.192; winnowing and string tiling (no graded score) error 0.163 and 0.149. 20 routes: attribution 0.065 / 0.107; earliest plausible 0.116, with abstention 0.070 (difference -0.005 [-0.008, -0.004]). The benefit of the unresolved outcome comes from abstaining by entry size, which works for any attribution with an order check.
- Online version: with delta = 0 identical to the offline labels (13 and 20 routes); with delta = 30 s on 20 routes 99.998% agreement (R11 online version 97.8%, streamed text 35%).
- FLoRA (`results/r12_flora.json`): 1.3% of X tokens (three students) are absent from the last snapshot; working span median 16 h (quartiles 1.5 h, 2.7 days; 84% at most four days); 38% of X tokens entered with the first snapshot and 82% in the last 24 h; U\* = 0 for 29.1% of students; naive overlap at least half of the proposal for 41.8% (n = 5), 36.0% (n = 8), 32.7% (n = 12). Pastes mostly X against mostly C, mixed models with a random intercept per student: later (p = 0.005), further from a reply (p < 0.001), length p = 0.06. Chatbot-panel selection before entry (`results/r12_audit_selection.json`): C 86% [74, 96], X 2% [0, 7].
- RelianceScope token entry rule, tau = 20 (`results/r12_rs.json`): T 22.6%, X 6.1%; 99.8% of these X tokens entered with the starter template when the editor was loaded.

## Decision list with delta > 0, sources of the risk-coverage difference, unresolved text (log entry R13)

No measure, class definition or parameter value changed, and no result with delta = 0 changed.

- Offline R2 now requires t_W + delta < t_R, as the decision list states (the code tested t_W < t_R; the two differ only for a shared n-gram whose first reply came within delta after entry). The online version now turns an S label into E when a reply within delta of entry follows a writer turn that contained the n-gram (R1 before R3). With R2 as written, online and offline labels agree on every benchmark token with delta = 30 s (R12: 99.998%, three tokens of reading highlights in streamed text). Streamed text 97.8% C and editor-first echo passed by the environment 7.8% C at delta = 30 s, as before (`results/r13_benchmark.json`).
- Remaining AURC difference between the attribution and earliest plausible attribution with abstention by entry size: 20 routes -0.0053, -0.0037 without the shared-source route (R3 decides its 3,136 tokens, the other method abstains on all of them) and -0.0034 without streamed text; 13 routes -0.0015, -0.0005 without the shared-source route.
- FLoRA (`results/r13_flora.json`): 37% of X tokens entered with the first snapshot (126 students, median first snapshot 642 tokens); the editor loaded no common text (56 of 33,486 such 8-grams occur in another student's first snapshot; 118 in one to four other proposals). Token-rule audit sample: X selected in the chatbot panel before entry 0 of 50, T 3 of 50.
- English Wikipedia (`src/r13_wiki.py`, `results/r13_wiki.json`; 6.4 million articles): share of class tokens in an 8-gram that also occurs in Wikipedia, C 0.9%, E 0.9%, S 36.9%, T 0.3%, X 0.8%.

Reproduce: `python -I src/r13_benchmark.py DATA R8_TOKEN_CLASSES && python -I src/r13_flora.py DATA && python -I src/r13_wiki.py SHARD_DIR` (R8_TOKEN_CLASSES: `git show af6bdb9:results/flora_token_classes.csv.gz > FILE`; `src/r13_wiki.py` needs `pyahocorasick` and `pyarrow` and downloads about 11 GB, one shard at a time).

## Reproduce

```bash
pip install -r requirements.txt
# put the Figshare files in DATA as dialogue.xlsx, writing.xlsx, proposal.xlsx, bio.xlsx, prior.xlsx, post.xlsx
python -c "import pandas as pd; pd.read_excel('DATA/writing.xlsx').to_pickle('DATA/writing.pkl')"
python -I src/run_s1.py --data DATA --out results/s1_stats.json
python -I src/run_s1_echo.py DATA results/s1_echo_stats.json
python -I src/run_s1.py --data DATA --out results/s1_stats_echo_removed.json --label results/Ue.npy
python -I src/align_check.py DATA
python -I src/run_s1b_provenance.py DATA
python -I src/run_s1.py --data DATA --out results/s1_stats_uprov.json --label results/Uprov_primary_0h.npy
# R1 also needs Figshare files 5.2 (trace) and 5.8 (annotation) saved as DATA/trace.pkl and DATA/annot.pkl
python -I src/r1_clock.py DATA
python -I src/run_r1.py DATA
python -I src/make_figs.py DATA && python -I src/make_pipeline_fig.py
# R2 (needs torch, transformers; models Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-0.5B-Instruct from Hugging Face)
python -I src/r2_label.py DATA
for st in d12 d3 merge; do python -I src/r2_detect.py DATA $st; done
python -I src/r2_stats.py && python -I src/r2_posthoc.py DATA
# R3
python -I src/run_r3_rs.py RS_DIR
python -I src/run_r3_flora.py DATA && python -I src/r3_d3swap.py
python -I src/make_fig_classes.py
# R4 (PROMPT_TXT: text of 'study materials/chatbot prompt.pdf', e.g. via pdftotext)
python -I src/r4_rs.py RS_DIR PROMPT_TXT
python -I src/r4_flora.py DATA && python -I src/r4_flora_tx_timing.py DATA
# R5 (r5_flora_para.py needs results/r2_scores.pkl from R2 and sentence-transformers)
python -I src/r5_clock.py DATA && python -I src/r5_flora_attr.py DATA && python -I src/r5_flora_grade.py DATA && python -I src/r5_flora_para.py DATA
python -I src/r5_rs.py RS_DIR PROMPT_TXT && python -I src/make_fig_classes_r5.py
# R6 (DATA also needs trace.pkl and annot.pkl as above)
python -I src/r6_flora.py DATA && python -I src/r6_rs.py RS_DIR PROMPT_TXT
# R7
python -I src/r7_flora.py DATA && python -I src/r7_rs.py RS_DIR PROMPT_TXT && python -I src/make_fig_classes_r7.py && python -I src/make_flow_fig.py
# R8 (R2_PASSAGES: results/r2_passages.pkl from src/r2_label.py)
python -I src/r8_flora.py DATA R2_PASSAGES && python -I src/r8_rs.py RS_DIR PROMPT_TXT && python -I src/r8_rs_llm.py RS_DIR PROMPT_TXT && python -I src/r8_rs_tobit.py RS_DIR PROMPT_TXT
```

Licence: MIT (code). FLoRA data remain under their CC BY 4.0 licence; RelianceScope data may be used and redistributed with citation of Jin et al. (2026), as stated by the dataset authors.

```
# R9
python -I src/r9_synthetic.py
python -I src/r9_flora.py DATA R2_PASSAGES
python -I src/r9b_flora_token_entry.py DATA R2_PASSAGES
python -I src/r9_rs.py RS_DIR PROMPT_TXT
```

```
# R10 (FLoRA writing log as DATA/writing.pkl, trace as DATA/trace.pkl)
python -I src/r10_benchmark.py DATA
python -I src/r10_online.py DATA RS_DIR
python -I src/r10_baselines.py DATA
python -I src/r10_rs.py RS_DIR PROMPT_TXT
python -I src/r10_flora_checks.py DATA
```

```
# R12 (R8_TOKEN_CLASSES: git show af6bdb9:results/flora_token_classes.csv.gz > FILE, the labels R11 read when it ran;
#      DATA/pastes.pkl: rows of Figshare files 5.2 with learning_action_label PASTE_TEXT_ESSAY)
python -I src/r12_benchmark.py DATA R8_TOKEN_CLASSES
python -I src/r12_flora.py DATA && python -I src/r12_audit_selection.py
python -I src/r12_rs.py RS_DIR PROMPT_TXT
python -I src/make_fig_rc_r12.py && python -I src/make_fig_classes_r12.py
```
