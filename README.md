# Provenance of GenAI text in student proposals (FLoRA): measurement, detection and grading

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24101340.svg)](https://doi.org/10.5281/zenodo.24101340)

v1.0.0: `10.5281/zenodo.24101340`. All versions: `10.5281/zenodo.24101339`.

Code, time-stamped analysis plans and statistics files for a study of how much of students' submitted work comes verbatim from a GenAI chatbot, whether dialogue behaviour can detect it, and whether rubric grades reflect it. Data: FLoRA Open Data IPS (Li et al., 2026; Figshare [10.6084/m9.figshare.32321499](https://doi.org/10.6084/m9.figshare.32321499); CC BY 4.0), 275 postgraduate students, 3,542 student–GPT-4o turns. No third-party data are redistributed. All analyses are exploratory.

## Order of work (see `git log`)

| Commit | Content |
|---|---|
| `PREREG_S1_flora.md` | Plan for Study 1, committed before any association was computed |
| `results/s1_stats.json` | Primary run as planned (naive uptake U) |
| Deviation D1 | Echo correction: exclude text the student pasted into prompts (`src/run_s1_echo.py`) |
| `src/align_check.py`, `PREREG_S1b_provenance.md` | Clock check and plan for provenance validation, committed before the provenance run |
| `results/s1b_provenance.json` | Provenance from the writing log; decision rule selects U_prov as the primary label |
| `results/s1_stats_uprov.json` | RQ2–RQ4 with U_prov |

## Main results (Study 1)

- Naive 8-gram overlap overstates uptake: median 0.21 of proposal tokens. Removing text the student had pasted into prompts gives a median of 0.027 (D1).
- Writing-log provenance: at the primary clock offset, 79.8% of the remaining overlap was in a chatbot reply before it appeared in the editor (93% at +1 h); 85% of these passages entered the editor in a single jump of 20 or more tokens. By the pre-specified rule (< 0.80), provenance-checked uptake U_prov is the primary label: median 0.019, 21.8% of students with at least a quarter of the proposal from chatbot text, 14.2% with at least half.
- Detection: no model separates high-uptake students (top quartile of U_prov) above chance; exposure-only AUC 0.522, dialogue behaviour 0.521, combined 0.512; gain over exposure −0.012 (95% CI −0.099 to 0.080).
- Grading: one SD of U_prov changes the proposal score by 0.07 points on a 15-point scale (95% CI −0.15 to 0.29), adjusting for prior knowledge, first language, degree and length; no association with self-reported gain in understanding (ρ = 0.03, n = 174).

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
```

Licence: MIT (code). Data remain under their own CC BY 4.0 licence.
