# Analysis plan R1 (revision 1)

Written and pushed before any of the analyses below were run. Before writing it we checked only data-structure facts: the system trace log (Figshare files 5.2) records chat-box keystrokes and PASTE_TEXT_ESSAY events with the pasted text; the dataset paper states that writing snapshots are recorded after each keystroke and that intent codes were assigned by human coders (average κ = 0.92); the annotation log (5.8) holds highlighted excerpts of the reading materials. We also matched Enter keystrokes in the chat box (trace clock, the same platform clock as the writing log) to dialogue ask times: 611 of 1,026 matched asks differ by less than 30 s and the median difference is about 2 s, so the two clocks agree and δ = 0 is a property of the data (`src/r1_clock.py`). These are time-stamped analysis plans, not registrations with a third party; all analyses remain exploratory.

## Changes to the measure

M1. **Time-aware echo correction.** An 8-gram is removed as the student's own only if it occurs in a prompt asked before the first reply that contains it. Report the number of 8-grams removed by the old (direction-blind) rule that first appeared in a reply and only later in a prompt.

M2. **Third sources.** Remove 8-grams that occur in (a) any highlighted reading excerpt in the annotation log, or (b) the final proposals of at least five other students (shared assignment-brief, template or reading phrases). Report how much each filter changes uptake.

M3. **Unresolved 8-grams.** 8-grams in the final proposal that never appear in any editor snapshot are treated as unresolved, not as chatbot-first. The earlier liberal rule is reported as sensitivity.

M4. **Clock.** δ = 0 (see above). +1 h reported as sensitivity.

**Primary measure R1:** U_R1 = coverage of the proposal by 8-grams that are in a chatbot reply, survive M1 and M2, and appear in the editor no earlier than their first reply (δ = 0). Sensitivity: n = 5 and n = 12; M2 off; M3 liberal; δ = +1 h.

## Validation

V1. Direct paste events: for students with PASTE_TEXT_ESSAY events, the share of U_R1 tokens whose 8-grams occur in logged pasted text, and the share of logged pasted text that is chatbot-first.
V2. Snapshot-based paste rule (net increase ≥ 20 tokens between consecutive snapshots): agreement with logged paste events for chatbot-first 8-grams.
V3. Chance: recompute uptake for each student against the replies of 200 random other students with M1–M2 applied to the comparison set; report each student's observed minus own null median, and the share of students above their own null 99th percentile.

## Research questions

RQ1. Distribution of U_R1 with 2,000-resample bootstrap intervals for the shares ≥ 25% and ≥ 50%; share above chance (V3); the same shares for U, Ue (old), U_prov (old). When uptake enters (share of U_R1 tokens first entering the editor in the last 24 h and last 2 h before the student's last snapshot) and where (first, middle, last third of the proposal).

RQ2 (secondary). Label: top quartile of U_R1 (report threshold and n positives). Feature sets: exposure (log chatbot words, log turns); coded intent shares + log turns; request features fixed now: share of prompts containing any of {write, draft, rewrite, paragraph, section, introduction, conclusion, proposal, improve, polish, paraphrase}, mean log words per reply, share of replies containing a markdown heading or bold line, share of turns in the 24 h before the student's last snapshot; all features combined. Protocol as Algorithm 4, with one convention: AUC of the seed-averaged out-of-fold predictions, with a 2,000-resample bootstrap interval; gains over exposure as before, same verdict rule. The claim is limited to the features tested.

RQ3 (primary for assessment). OLS of score on standardised U_R1, HC3 errors: (a) unadjusted; (b) pre-task covariates only (prior knowledge, GenAI familiarity, English first language, degree); (c) (b) + log proposal length + log chatbot words. Years of experience is dropped (60% missing). Nonlinearity: mean score of students with U_R1 ≥ 0.5 minus the rest, bootstrap interval; mean score by uptake quartile. Understanding: Spearman as before; uptake of survey responders vs non-responders. Criterion-level scores are not in the release; this is reported as a limitation.

RQ4 (descriptive). Mean, median and share ≥ 25% of U_R1 by first language (English / other), bootstrap intervals; no model-based flag comparison.

## Not done

No change of n-gram length, thresholds or keyword list after seeing R1 results.
