# Analysis log entry R26

Written on 24 November 2026, after the analysis below had run, in preparation of the twelfth version of the manuscript. Nothing in the attribution, its rules or its parameters changed.

## F. Four FLoRA students absent from the release (src/r26_flora_missing.py)

The data descriptor of FLoRA Open Data IPS (Li et al., 2026, Scientific Data, doi:10.1038/s41597-026-08054-w) reports 279 retained students; every file of the public release covers 275, and the description does not say why four are absent. Nothing is known about the four, so every row of the identified-range ladder of R23 (F-51) is recomputed over 279 writers with the four at the least favourable value for each end: for a lower end they count as writers with no chatbot text, for an upper end as writers whose whole proposal is chatbot text. Input: `results/r23_flora_students.csv`, released with R23. Output: `results/r26_flora_missing.json`.

- F-55. Share of writers at or above half (release, n = 275 → worst case, n = 279): lower end C 20.0% → 19.7%; C + E-after 23.6% → 23.3% (lower) and 24.7% (upper); upper end under A1–A3, N, B, R (C + E-after + X-with-C) 37.5% → 38.4%; without B 81.8% → 82.1%; without B and A1 92.0% → 92.1%. The identified range [20.0%, 37.5%] becomes [19.7%, 38.4%] and the range without B [20.0%, 81.8%] becomes [19.7%, 82.1%].
- F-56. Mean share at the lower end (C): 23.6% → between 23.3% and 24.7%. At 25% and 75% each end moves by at most 1.4 points.
- The counts at n = 275 reproduce those of R23 F-51 exactly (55, 103, 225 and 253 writers at or above half).

## Withdrawal of the NIRVANA results

NIRVANA (Jelson et al., 2026) was the hold-out set of R22 and R23. Its OSF project 3a8uh became private on 22 November 2026, the access request through OSF was not granted, and its files state no licence. Because permission to publish results computed on the data could not be established, the NIRVANA results are withdrawn:

- removed from this release: `src/r22_nirvana.py`, `src/r22_nirvana_posthoc.py`, `src/r23_nirvana_agg.py`, `results/r22_nirvana.json`, `results/r22_nirvana_posthoc.json`, `results/r23_nirvana_agg.json`;
- in `ANALYSIS_LOG_R22.md` (results N-1 to N-9 and the line of class shares) and `ANALYSIS_LOG_R23.md` (N-10 to N-12) each NIRVANA result is replaced by a withdrawal note; the plan of R22, the FLoRA and benchmark results of R22 (F-50, B-47) and everything else are unchanged;
- the manuscript and its Supplementary Material report no NIRVANA result.

The copy of the data downloaded on 22 November 2026 has not been used since. Releases archived before v1.26.0 are unchanged. No rule, parameter or FLoRA result was chosen or changed on the basis of the NIRVANA analyses: the rules and parameters were frozen at R10. CoAuthor (R24, R25) is the hold-out set.
