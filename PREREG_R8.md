# Plan R8: measurement error, rule order, retyped code and precision of the null result

Committed and pushed before any R8 analysis was run. All R1–R7 results are known, so every analysis below is exploratory and post hoc. No analysis changes the primary measure U\*, the class definitions of plan R4 or the parameter values used in the manuscript; R8 reports how much the interpretation of the results depends on them.

## J. FLoRA

**J1. Misclassification of uptake.** Part of the untraced class X may be chatbot text edited outside the platform or taken from another AI tool. Per student, compute the token shares C, X, X_c (X tokens whose 8-grams entered the editor in the same snapshot transition as at least one chatbot-first 8-gram) and X_o = X − X_c. Regress the grade (OLS, HC3) on the standardised composite C + λ·X for λ ∈ {0, 0.25, 0.5, 1}, and on C + λ·X_c for λ ∈ {0.5, 1}; and fit a two-component model with standardised C and X_c entered together. Each model is fitted unadjusted (model a) and with the pre-task covariates of model (b), median-imputed as in the primary analysis. Report slopes per SD with 95% intervals and p values.

**J2. Equivalence test.** Two one-sided tests (TOST) for the primary unadjusted slope and for the J1 composites, with equivalence bounds of ±0.25 grade SD per SD of uptake (±0.5 points with the observed grade SD of 2.0). The bound is chosen here, after the estimates are known, and is declared post hoc. Also report the 90% interval and the smallest symmetric bound at which equivalence is declared at α = 0.05.

**J3. Order of the E and S rules.** Count the 8-grams of the final proposals that are labelled S but first appeared in the editor before the first reply containing them (editor-first echo that the S rule takes precedence over). Relabel them E and report the number of tokens that change class and the class means.

**J4. Students contributing to the detector comparison.** Number of students contributing chatbot, typed and untraced paragraphs under the plan R4 labels (paragraph labels from `src/r2_label.py`), and the largest share of typed paragraphs contributed by one student.

**J5. Token-level provenance annotation.** Export the five-class label of every token of the 275 final proposals (`results/flora_token_classes.csv.gz`: user_id, token index, token, class) and the per-student class shares. Tokens are the lower-cased alphanumeric tokens of the released proposals (CC BY 4.0), so the file is a derived work under the same licence.

## K. RelianceScope

**K1. Segments coded "write over the response".** For this code and for "copy the response": share of segments with any uptake and one-against-rest AUC of segment uptake under (a) n = 5, 8, 12 with alphanumeric tokens; (b) the two code-aware lexers at n = 8 and n = 5; (c) a wider segment window, ending at the start of segment j + 2, and an open window (any later editor entry); (d) a label-free count: reply n-grams of the segment present in the final code, whatever their class. Describe the write-over segments with no uptake at n = 8: share whose replies contain a code block, share with any reply 8-gram in the final code, share with any reply 5-gram entering the editor within the segment window.

**K2. Order of the E and S rules.** As J3, for the starter template.

**K3. Agreement ceiling under label noise.** For the binary criterion copy against the rest (prevalence 119/427), compute by simulation (10,000 draws) the expected AUC of a score that separates the true classes perfectly when the observed labels are misclassified non-differentially with probability q ∈ {0.05, 0.10, 0.15, 0.17}; q = 0.17 is the per-coder error rate implied by 71.6% agreement between two coders on a binary code, an upper value because the reported agreement covers all response-use modes.

**K4. Post-test models without the post-treatment covariate.** Task progress and minutes on the activity are measured after and partly because of chatbot use. Refit the plan R5 OLS and Tobit models with the pre-test plus the pre-activity questionnaire scores (self-efficacy, intrinsic and extrinsic motivation, metacognition) in place of task progress, for the three uptake measures. These are measured before the activity in the RelianceScope design.

## Addendum (committed before K5 was run; K1–K4 were running, J1–J5 not yet started)

**K5. Comparison with the LLM classifiers released with RelianceScope.** The RelianceScope release contains a benchmark of 150 interaction segments with the human response-use mode (passive, active, constructive) and the outputs of 24 LLM configurations (`llm-evaluation/`). Passive response use comprises the detailed codes copy, write over and read and move on. Match the benchmark segments to the coded segments by student and start time. On the matched segments, for passive against the rest: precision, recall and F1 of the rule "segment uptake > 0" (n = 8, alphanumeric tokens), the AUC of segment uptake, and the precision, recall and F1 of each LLM configuration recomputed on the same segments. Report the same for the narrower class copy or write over, for which the LLM outputs give no label, only for segment uptake. Intervals: 2,000 bootstrap resamples of students. Descriptive; no decision rule.
