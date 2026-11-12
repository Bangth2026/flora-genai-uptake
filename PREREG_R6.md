# Plan R6: analyses requested in peer review

Committed and pushed before any R6 analysis was run. All R1–R5 results are known, and the reviewer's comments are known, so every analysis below is exploratory and post hoc with respect to the earlier plans. No analysis below changes the primary measure U\*, the class definitions of plan R4 or the parameter values used in the manuscript; R6 reports how much the results depend on them.

## G. FLoRA

**G1. Degraded logs (reviewer comment 1, question 1).** Three reduced versions of the logs.
- (a) Dialogue only, no editor snapshots. Labels: E if the n-gram occurred in a student turn before the first reply containing it; S as in R4; C⁻ if it occurs in a reply and is not E or S (no order check against the editor); every other n-gram is unattributed (no T/X split is possible). Report median C⁻, share of students with C⁻ ≥ 0.5, Spearman ρ with U\*, the share of R4 E tokens recovered as E, and the share of R4 C tokens labelled C⁻.
- (b) Thinned snapshots. Keep a snapshot only if at least Δ ∈ {5, 15, 60, 300} s have passed since the last kept snapshot; always keep the final one. Recompute all labels with τ = 5 and τ = 20. Report mean class shares, median U\*, the share of tokens whose class differs from the full-resolution R4 labels, and, among tokens labelled T or X in both versions, Cohen's κ for the T/X split.
- (c) No clipboard events. No attribution rule uses clipboard events, so the classes are unchanged; only validation V1, V2 and V4 is lost. Stated, no analysis.

**G2. Parameter choices (reviewer comment 5).**
- (a) n. The V3 chance baseline at n = 5, 8 and 12: median coverage of a student's proposal by the replies of a random other student (200 draws per student, shared-source n-grams excluded), and the share of students whose own-reply coverage exceeds their null 99th percentile.
- (b) τ. Mean T and X shares and median U\* at τ ∈ {1, 2, 5, 10, 20, 50}.
- (c) Effect on conclusions. For each parameter variant already reported (n, δ, peer threshold, projection, entry rule) and for the new τ variants: median U\*, share ≥ 0.5, mean X share and the unadjusted OLS grade slope per SD of uptake with HC3 interval.

**G3. Subdivision of X (question 2).** Descriptive split of X tokens by entry evidence: (i) the 8-gram entered with a logged paste within 10 s (68 students with logged pastes); (ii) the 8-gram entered in the same transition as at least one C 8-gram of the same student; (iii) size of the entering segment, 6–50 tokens against more than 50 tokens. No source is assigned.

**G4. Origin of logged pastes (reviewer comment 3).** For each logged paste with at least eight tokens, the share of its 8-grams that occur (a) in a reply of the same student before the paste, (b) in a student turn before the paste, (c) in an earlier editor snapshot of the student, (d) in none of these. Pastes are classed as reply-sourced (≥ 80% in a), internal (≥ 80% in c), external (≥ 80% in d) or mixed. For the C tokens confirmed in V1, report the share whose covering paste was reply-sourced, mixed or other.

**G5. Precision of the outcome estimates (reviewer comment 4).** Minimum detectable effect at 80% power and α = 0.05 as 2.80 × SE of the HC3 OLS slope, for FLoRA models (a)–(c) and the RelianceScope OLS models. SD of U\* in each dataset, and each slope re-expressed per 10 percentage points of uptake. For the RelianceScope Tobit model, simulation-based power: post-test generated from the fitted latent model (observed pre-test and U\*, fitted intercept, pre-test slope and σ), censored at 10, with slope per SD of U\* ∈ {−0.25, −0.50, −0.75, −1.00}; 1,000 simulations each.

## R. RelianceScope

**G6. Reference n-grams removed (question 3).** Already reported in plan R5 (bound (a) as predictor, with and without task progress). No new analysis; the R5 values are cited.

**G7. Dialogue only for code.** As G1(a), using the code attached to each student message as the only editor evidence (no snapshots): C⁻, its Spearman ρ with U\*, and its mean.

## D. Demonstration

**G8.** A self-contained implementation of Algorithms 1–2 and 4 (`demo/provenance.py`) with synthetic logs and an end-to-end script, so the method can be run without either dataset. The demo reproduces no study result.
