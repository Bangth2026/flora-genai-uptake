# Analysis plan R2 (revision 2)

Written and pushed before any detector score was computed. Before writing it we checked only that the three open models listed below download and load on CPU, and that final proposals are split into paragraphs by newline characters (median 44 newlines per proposal). These are time-stamped analysis plans, not registrations with a third party; all analyses remain exploratory. The primary uptake measure, U* (= U_R1 in `PREREG_R1.md`), and all its parameters are unchanged.

## D. Open zero-shot detectors (new RQ5)

Question: do open, zero-shot AI-text detectors separate proposal text that the process logs show was taken verbatim from the chatbot from text that the logs show was composed in the editor, and do they flag composed text of students whose first language is not English more often?

### Units and labels (fixed now)

- Passage = one paragraph of the final proposal (split on newlines), keeping paragraphs with at least 40 word tokens (same tokenizer as the uptake measure).
- Chatbot passage: at least 80% of the passage's tokens are covered by the student's chatbot-first 8-grams (the U* set).
- Composed passage: 0% coverage by chatbot-first 8-grams and less than 5% coverage by 8-grams of the student's own logged PASTE_TEXT_ESSAY events, and no 8-gram from any chatbot reply of that student.
- Other passages (mixed) are excluded from the primary contrast and counted.
- If either class has fewer than 30 passages, the length floor drops to 25 tokens (one step only).
- Passages are truncated to the first 256 model tokens.

### Detectors (fixed now)

D1. Mean token log-likelihood under Qwen2.5-1.5B (higher = more machine-like).
D2. Fast-DetectGPT analytic criterion (Bao et al., 2024) with Qwen2.5-1.5B as both sampling and scoring model.
D3. Binoculars score (Hans et al., 2024) with Qwen2.5-0.5B as observer and Qwen2.5-0.5B-Instruct as performer (lower = more machine-like; sign flipped for AUC).

All run in float32 on CPU, no fine-tuning, no threshold tuned on the FLoRA labels except as stated below.

### Analyses

D-a. Passage AUC (chatbot vs composed) for each detector, with a 2,000-resample bootstrap interval that resamples students (cluster bootstrap).
D-b. Operating points: thresholds at which each detector flags 50% and 80% of chatbot passages; report the share of composed passages flagged at each (false-flag rate), overall and by first language (English / other), with cluster-bootstrap intervals for the difference (other − English).
D-c. Within-student contrast: for students with at least one passage of each class, the mean paired difference of detector scores (chatbot − composed), cluster bootstrap.
D-d. Composed-passage scores by first language: mean difference (other − English), cluster bootstrap. Passage length (log tokens) is reported alongside, because detector scores depend on length.
D-e. Proposal level: Spearman correlation between each detector's score on the first 512 model tokens of the whole proposal and U*; AUC for U* ≥ 0.5 vs U* < 0.05.

Wording rule fixed now: AUC interval lower bound ≥ 0.8 is described as clear separation; point estimate 0.6–0.8 as partial; below 0.6 as weak. A false-flag difference by first language is described as present only if its interval excludes zero.

## S. Sensitivity of the third-source filter

Recompute U* with the peer-proposal threshold k = 3, 5 (primary) and 10 other students; report median, shares ≥ 25% and ≥ 50%, and the unadjusted grade slope (model a, HC3) for each.

## Not done

No other detectors, models, passage rules or thresholds after seeing detector scores. Commercial detectors are not used: their model versions change without notice, so results could not be reproduced from the released code.

## Deviations

R2-1 (before any detector score was computed). Loading Qwen2.5-1.5B in float32 exceeded the 7 GB memory of the analysis machine and the process was killed before scoring its first passage. Its weights are therefore loaded in bfloat16; logits are cast to float32 before the D1 and D2 computations. The two 0.5B models stay in float32.
