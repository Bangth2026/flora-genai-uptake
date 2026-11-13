# Analysis log entry R12

Written after all earlier results (S1 to R11) were known. Unlike entries R1 to R11, this entry was written after its analyses had run: the scripts were developed and run on reduced sizes and then in full in one session, and the entry records what was done and why. All analyses are exploratory. No measure, class definition or parameter value changes; the primary attribution, U\* and all class shares of R11 are unchanged.

## Description of the rules and of the online version

D-1. Order of the rules. The code has always applied both echo rules before the shared-source rule (`demo/provenance.py`, `src/r8_rs.py`, `src/r11_flora.py`, `src/r11_benchmark.py`: chatbot-first requires that the n-gram is not shared; otherwise echo, turn first or editor first, is tested before shared source). The decision list in the article is written in this order: R1 echo (turn first), R2 echo (editor first), R3 shared source, R4 chatbot-first, R5 no evidence. R2 and R4 are decided by the order of the same two times and never overlap. No n-gram of FLoRA or RelianceScope meets both R2 and R3 (log entry R10, addendum), so no label changes.

D-2. Online version with delta > 0 (`demo/provenance_online.py`). The R11 version decided chatbot-first only at entry, from the replies already seen, so a reply that arrived within delta after an n-gram entered the editor left the n-gram T or X, whereas the offline rule gives C. The online version now treats a label as provisional for delta seconds after entry: a reply within that window turns a T or X label into C (or E if a writer turn contained the n-gram before the reply), as the offline rule does; later replies can only turn S, T or X into E. With delta = 0 (the primary setting for both datasets) the window is empty and every result of R10 and R11 is unchanged.

## B. Benchmark (src/r12_benchmark.py)

The simulated students are those of R11 (seed 2100, 13 routes; seed 2200, 20 routes), rebuilt with the definitions of `src/r11_benchmark.py`, which the script executes, and with the token-level FLoRA labels that R11 read when it ran (the file released with R8; R11 re-exported it after its benchmark had run). A check reproduces the R11 F1 of the chatbot-first class (0.916).

B-10. Selective prediction with a stated confidence for every method, decision "chatbot text":
- attribution: as in R11 (abstains on X; otherwise the number of covering n-grams that carry the token's label);
- earliest plausible attribution: the number of covering n-grams whose earliest author agrees with the token's decision (R11 gave it no ordering);
- earliest plausible attribution with abstention by entry size: abstains on tokens it does not mark as chatbot text that entered in an insertion of more than tau tokens and are covered by no reply n-gram, the condition that defines X;
- dialogue only and naive overlap: distance of the share of covering n-grams found in the matching set from one half (as R11), and the same with abstention by entry size;
- direction-blind removal: as dialogue only, with reply n-grams not in any turn;
- paste-log rule: abstains on large insertions that it does not call chatbot text ("unknown"); typed tokens confidence 0.5, chatbot tokens the share of the insertion's n-grams in earlier replies minus one half;
- winnowing and string tiling give a binary mark without a graded score; only their error when every token is decided is reported.
For each: AURC, error when every token is decided, coverage and risk before abstention; paired bootstrap (1,000 resamples of simulated students) of the AURC difference between the attribution and earliest plausible attribution with and without abstention, dialogue only and naive overlap.

B-11. Online against offline labels with delta = 0 (13 and 20 routes) and delta = 30 s (20 routes), with the R11 online version as a comparison; transitions of n-gram labels within and after delta of entry.

## F. FLoRA (src/r12_flora.py)

F-6. Unresolved tokens of the submitted proposal that do not occur in the last editor snapshot (entry size infinite in Algorithm 2).
F-7. Working span (first to last editor snapshot) per student; share of C, T and X tokens entering in the first snapshot, the first hour, the first and last quarter of the span, and the last 24 h, overall and for students whose span exceeds four days.
F-8. Distribution of U\*: share of students with U\* = 0, quartiles, skewness.
F-9. Naive 8-gram overlap at n = 5 and 12 (median, share at or above one half, student bootstrap interval).
F-10. Logged pastes mostly C against mostly X: linear mixed models of log length, log hours before the last snapshot and log seconds since the last reply with a random intercept per student, and student-cluster bootstrap intervals of the difference in medians (R11 reported Mann-Whitney tests that treated pastes as independent).

F-11. Share of the audited passages of R9 that were selected in the chatbot panel before entry, by class, with student-resampled intervals (`src/r12_audit_selection.py`).

## K. RelianceScope (src/r12_rs.py)

K-4. Class shares under the token entry rule (primary for FLoRA) and the n-gram entry rule, tau = 5 and 20; composition of tokens that entered in insertions of more than tau tokens, and whether unresolved tokens entered with the first snapshot (the starter template loaded into the editor) or later.
