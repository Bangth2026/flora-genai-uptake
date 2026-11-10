# Analysis plan S1b: validating the uptake measure with the writing log

Written after the Study 1 primary run and Deviation D1, and before any provenance statistic was computed. The only writing-log analysis run beforehand is the clock check in `src/align_check.py`, which uses text students pasted from the editor into prompts and no uptake or outcome variable. Its result: at offset 0, 68% of such text is already in the editor at ask time; at +1 h, 89%; the share plateaus at 94% from +4 h. We take offset 0 as primary (both clocks appear to be UTC; there is no sign of a 10-hour shift) and report +1 h and +4 h as sensitivity.

## Procedure

For each student, G = the 8-grams of the final proposal that occur in the student's chatbot responses and not in the student's own prompts (the grams counted by the echo-corrected measure Ue). For each gram in G:
- t_bot = time of the earliest chatbot response containing it;
- t_ed = time of the earliest writing-log snapshot containing it (snapshots in time order; normalised as in Study 1).

A gram is **chatbot-first** if it never appears in the editor or t_bot ≤ t_ed (plus offset), and **editor-first** otherwise. U_prov = share of proposal tokens covered by chatbot-first grams.

## Statistics and decision rule

- Pooled token-weighted share of Ue-covered tokens that are chatbot-first. Ue is treated as a valid measure of uptake if this share is at least 0.80 at the primary offset; between 0.60 and 0.80, Ue is reported with this caveat and U_prov becomes the primary label; below 0.60, Ue is not used.
- Spearman correlation between Ue and U_prov.
- Paste evidence: share of chatbot-first grams that enter the editor in a single snapshot transition adding at least 20 tokens.
- RQ2–RQ4 of PREREG_S1_flora.md repeated with U_prov as the label.

## Outcome of the decision rule (recorded after the run)

Share of Ue-covered tokens that are chatbot-first: 0.798 at the primary offset (0.930 at +1 h, 0.947 at +4 h). Under the rule above, U_prov becomes the primary label; RQ2–RQ4 were repeated with it (`results/s1_stats_uprov.json`).
