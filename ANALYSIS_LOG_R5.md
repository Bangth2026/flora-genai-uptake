# Plan R5: checks of the attribution's assumptions and robustness of the outcome analyses

Committed before any R5 analysis was run. R1–R4 results are known. All analyses are exploratory; no result below changes the primary measure U\* or the class definitions of plan R4 unless stated as a decision rule.

## F. FLoRA

**F1. Within-document moves in X.** An X 8-gram first appears in a snapshot reached by an edit of more than J = 5 tokens. Such an edit can be a paste from outside the logs, or a rearrangement of text already in the document (cut and paste, drag and drop, undo of a deletion). For every transition between consecutive snapshots, compute the inserted and replaced segments of the token alignment. Keep a cumulative set of 8-grams present in any earlier snapshot of the same student. For each X 8-gram, take the inserted segment that contains it at entry and compute the share of that segment's 8-grams already present earlier.
- Report the share of X tokens whose entry segment was at least 50% previously present ("moved"), and the share at most 50% ("new"), with 20% and 80% as sensitivity thresholds.
- Decision rule: if moved X exceeds 10% of X tokens, report a six-class variant in which moved text forms its own class M, and state the untraced share without it in the abstract.
- Also report the same split for T, the distribution of inserted tokens per transition (all transitions) and the distribution of time between snapshots.
- Segment-local entry rule: entry mode decided by the size of the inserted segment that contains the n-gram rather than by the whole transition. Report class shares.

**F2. Students without logged pastes.** Compare students with and without PASTE_TEXT_ESSAY events: dates of activity relative to the first logged paste in the cohort, number of large insertions, and X share. Describe whether the absence looks like a logging gap.

**F3. Token-level projection.** U\* and class shares under (a) the R4 projection (C > E > S > T > X), (b) E before C (E > C > S > T > X), (c) majority label of covering n-grams (ties broken by the R4 order). Report the share of tokens whose class changes. Same for RelianceScope.

**F4. Clocks.** Full distribution of ask time minus nearest Enter keystroke (log scale), the share of differences within ±5 min of ±1 h, ±10 h and ±11 h, and the share of C n-grams whose reply and editor times lie within 10 s of each other (ties go to C under δ = 0).

**F5. Origin of E.** Share of E 8-grams whose first editor appearance came through an edit of more than J tokens, and share of prompt-first E 8-grams whose prompt contained at least 50 tokens.

**F6. Timing baseline.** Share of C, E, T and X tokens that entered the editor in the 24 h and 2 h before the student's last snapshot, and relative to the cohort's latest final snapshot time used as a deadline proxy.

**F7. Grade.** Number of missing values per covariate; models (b) and (c) of R1 with (i) multiple imputation by chained equations (20 imputations, Rubin's rules) and (ii) missing-indicator coding; grade distribution and share at 15; Tobit censored at 15; U\* as rank (percentile) and as log(U\* + 0.01), and a natural spline with 3 df (joint Wald test); grade on X and T shares (per SD), unadjusted and adjusted.

**F8. Detectors.** Paragraph labels with alternative thresholds: chatbot ≥ 70% or ≥ 90% C; typed and untraced ≥ 80% or ≥ 95%. Report AUCs. The first-language comparison within typed paragraphs is reported descriptively (means and counts, no interval).

**F9. V5, size-matched and semantic.** Containment in one random other student's replies (20 draws, averaged) instead of 20 students pooled. Sentence-embedding check: for each typed and untraced paragraph, maximum cosine similarity to sentences of the student's own replies and to sentences of one random other student's replies (all-MiniLM-L6-v2).

## R. RelianceScope

**R1. Code-aware tokenisation.** Two lexers: (a) code tokens keeping operators and punctuation, identifiers kept whole (letters, digits, _, $, -, @, : joined); (b) as (a) with identifiers and literals normalised to a placeholder, keywords and HTML/Vue directives kept. n = 8 and n = 12. Report U\*, null median, chance-corrected U\*, bound (a), and the copy AUC for each.

**R2. Outcome models with bounded measures.** OLS, Tobit (censored at 10) and ordered logit of the post-test on the pre-test and, in turn, standardised U\*, chance-corrected U\*, and bound (a). Adjust for task progress: final-code coverage of the reference solution (share of reference 8-grams in the final code) and activity minutes. Report all.

**R3. Copied strings.** Share of copied strings found in an earlier reply, in an earlier snapshot of the student's own code, in both, or in neither, by length (< 8 tokens, ≥ 8 tokens).

**R4. Ties and segment-local entry.** As F3 and F1 (segment-local rule) for code.

## Reporting

All R5 results go to `results/r5_*.json`. The order of plans R1–R5 is recorded by commit hashes and push times listed in the README.
