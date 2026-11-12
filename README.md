# Five-class provenance of chatbot text in student writing (FLoRA) and programming (RelianceScope)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24101339.svg)](https://doi.org/10.5281/zenodo.24101339)

All versions: concept DOI 10.5281/zenodo.24101339.

Code, time-stamped analysis plans and statistics files for a study that attributes each token of students' submitted work to five provenance classes (chatbot-first, echo of the student's own text, shared source, typed, untraced insertion) from dialogue, editor and clipboard logs, and relates verbatim uptake to grades, a post-test, dialogue features and AI-text detectors. Data: FLoRA Open Data IPS (Li et al., 2026; Figshare [10.6084/m9.figshare.32321499](https://doi.org/10.6084/m9.figshare.32321499); CC BY 4.0; 275 postgraduate students) and RelianceScope (Jin et al., 2026; 79 students; used with the authors' permission, not bundled here). All analyses are exploratory.

## Order of work (see `git log`)

| Commit | Content |
|---|---|
| `PREREG_S1_flora.md` | Plan for Study 1, committed before any association was computed |
| `results/s1_stats.json` | Primary run as planned (naive uptake U) |
| Deviation D1 | Echo correction: exclude text the student pasted into prompts (`src/run_s1_echo.py`) |
| `src/align_check.py`, `PREREG_S1b_provenance.md` | Clock check and plan for provenance validation, committed before the provenance run |
| `results/s1b_provenance.json` | Provenance from the writing log; decision rule selects U_prov as the primary label |
| `results/s1_stats_uprov.json` | RQ2–RQ4 with U_prov |
| `PREREG_R1.md`, `src/r1_clock.py` | Plan R1 and clock check, pushed before the R1 analyses |
| `results/r1_stats.json` | R1 analyses (`src/run_r1.py`) |
| `PREREG_R2.md` | Plan R2 (open AI-text detectors, third-source threshold), pushed before the R2 analyses; deviations R2-1 and R2-2 recorded |
| `results/r2_stats.json`, `results/r2_sens.json` | R2 analyses (`src/r2_label.py`, `src/r2_detect.py`, `src/r2_stats.py`); post hoc check `results/r2_posthoc.json` |
| `PREREG_R3.md` | Plan R3 (five provenance classes C/E/S/T/X; second dataset with human response-use codes and post-test; FLoRA typed/untraced reanalysis declared post hoc), pushed before the R3 analyses |
| `results/r3_rs_stats.json`, `results/r3_flora_stats.json`, `results/r3_d3swap.json` | R3 analyses (`src/run_r3_rs.py`, `src/run_r3_flora.py`, `src/r3_d3swap.py`) |
| `PREREG_R4.md` | Plan R4 (edit-size entry rule; T/X validation against logged pastes; chance baseline and bounds for code; length-controlled human-code validation; censored post-test model), pushed before the R4 analyses |
| `results/r4_rs_stats.json`, `results/r4_flora_stats.json`, `results/r4_flora_tx_timing.json` | R4 analyses (`src/r4_rs.py`, `src/r4_flora.py`, `src/r4_flora_tx_timing.py`) |

## Main results (plan R1, primary measure U*)

U* is the share of the final proposal covered by 8-grams that are in a chatbot reply, were not supplied by the student in an earlier prompt (time-aware echo correction), do not occur in reading highlights or in at least five other proposals, and were in a reply no later than their first appearance in the editor. Clocks aligned (chat-box Enter keystrokes vs ask times: median difference 2.2 s, `results/r1_clock.json`).

- A direction-blind echo correction (v1.0) removed 25,994 proposal 8-grams, of which 9,533 (36.7%) had first appeared in a chatbot reply.
- Logged paste events (68 students) contain 97.1% of U*-covered tokens.
- Median U* 0.048; 30.5% of students take at least a quarter of the proposal from chatbot replies, 20.0% at least half; 69% of uptake tokens enter in the last 24 h before the last snapshot.
- Grade per SD of U*: 0.02 points on 0–15 (95% CI −0.21 to 0.24); mean grade by uptake quartile 11.9, 11.8, 11.8, 11.7.
- Detection of the top quartile: exposure AUC 0.615 (0.538–0.693); coded intent and request features add nothing detectable (all gains inconclusive).

## Five provenance classes (plan R3)

Each token of the final artefact is labelled C (chatbot-first), E (echo of the student's own text), S (shared source), T (typed in the editor) or X (untraced insertion). U* is the share of class C.

| Mean share of tokens | C | E | S | T | X |
|---|---|---|---|---|---|
| FLoRA proposals (n = 275) | 23.6% | 13.6% | 0.2% | 12.8% | 49.8% |
| RelianceScope code (n = 79) | 23.4% | 5.0% | 42.9% | 28.6% | 0.1% |

RelianceScope validation: segment uptake vs human codes "copy / write over the response" AUC 0.759 [0.713, 0.799]; U* vs perceived reliance ρ = 0.65. Post-test on pre-test and U*: −0.32 per SD [−0.71, 0.06] (58% at ceiling). FLoRA detectors, chatbot vs typed paragraphs: AUC 0.805 (D1), 0.665 (D2), 0.710 (D3); untraced paragraphs score like chatbot paragraphs.

Plan R4 (entry mode by edit size): FLoRA T 12.4%, X 50.2%; 89.7% of X 8-grams but 3.0% of T 8-grams entered through a logged paste. RelianceScope chance-corrected U* 13.2% (raw 23.4%); human-code AUC for "copy the response" 0.762, gain over a reply-length model 0.187 [0.139, 0.238]; post-test Tobit −0.75 per SD of U* [−1.37, −0.12].

The RelianceScope data (Jin et al., 2026; [10.1145/3774398.3811612](https://doi.org/10.1145/3774398.3811612)) are used with the permission of the dataset authors, who allow use and redistribution provided the paper is cited. They are not bundled here: obtain them from the authors' Open Science Framework repository and place them in RS_DIR. This repository holds only code and aggregate RelianceScope statistics; `src/run_r3_rs.py` also writes a student-level file (`results/r3_rs_students.csv`), which is excluded from version control and must not be shared.

## Open AI-text detectors (plan R2)

Paragraphs of the final proposals (≥ 40 tokens) labelled from the logs: 147 chatbot (≥ 80% covered by chatbot-first 8-grams) and 618 composed (no reply 8-gram, < 5% logged paste).

| Detector | AUC [95% CI] | False flags at 50% / 80% detection |
|---|---|---|
| D1 mean log-likelihood, Qwen2.5-1.5B | 0.649 [0.586, 0.708] | 26.7% / 60.8% |
| D2 Fast-DetectGPT, Qwen2.5-1.5B | 0.533 [0.471, 0.596] | 46.8% / 75.2% |
| D3 Binoculars, Qwen2.5-0.5B / 0.5B-Instruct | 0.573 [0.501, 0.638] | 41.4% / 73.8% |

Composed paragraphs of students with a first language other than English scored as more machine-like on all three detectors (intervals exclude zero). Third-source threshold k = 3, 5, 10 leaves U* and the grade slope unchanged.

All numbers: `results/r1_stats.json`, `results/r2_stats.json`. Earlier versions of the measure: `results/s1_*.json`, `results/s1b_provenance.json`.

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
```

Licence: MIT (code). FLoRA data remain under their CC BY 4.0 licence; RelianceScope data may be used and redistributed with citation of Jin et al. (2026), as stated by the dataset authors.
