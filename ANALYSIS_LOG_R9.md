# Analysis log entry R9

Written after all earlier results (S1 to R8) were known. All analyses below are exploratory and post hoc. None changes a measure, a class definition or a parameter value of the primary attribution (n = 8, tau = 5 for FLoRA and 20 for RelianceScope, delta = 0, peer threshold 5, token priority C > E > S > T > X).

## Synthetic logs with known routes (src/r9_synthetic.py)

S-1. Simulated students write documents from blocks with known routes: typed (T); copied from a reply, pasted or retyped (C); own draft sent in a prompt and pasted back from the reply (E1); typed text later repeated by the chatbot (E2); typed, deleted, sent to the chatbot and pasted back (E3); shared source (S); pasted from outside the logs (X). The attribution of demo/provenance.py is applied unchanged. Token-level precision, recall and confusion per class are reported for a baseline and under perturbations: clock offset between dialogue and editor (+10 s, +60 s, +1 h, -60 s, -1 h), snapshot thinning (5, 15, 60, 300 s with tau = 5 and 20) and partial paraphrase of pasted chatbot text (every 12th, 8th or 4th token replaced). 300 students per condition, seed 2090.

## FLoRA (src/r9_flora.py)

F-1. Echo by entry mode and route: share of E tokens whose covering E n-grams entered the editor in small edits (E-T) or larger insertions (E-X), and prompt-first versus editor-first E. Students with any E-X and their U*. Share of students at or above 50% under C + E-X, an upper variant that counts echo entering as large insertions as chatbot text.
F-2. Runtime of Algorithms 1 and 2 per student and in total on one CPU core, with the number of snapshots.
F-3. Students whose chat-box keystrokes were logged (the clock-check group) against the rest: median U*, share at or above 50%, mean X, grade slope.
F-4. Snapshot thinning: share of all tokens that change class (not only within T/X) at 5, 15, 60 and 300 s with tau = 5 and 20.
F-5. Entry threshold: for tau = 1, 2, 3, 5, 10, 20, 50, the share of T and of X 8-grams that entered with a logged paste (68 students), and the share of tokens changing between T and X relative to tau = 5. Paste window of 5, 10, 30 and 60 s for the tau = 5 version.
F-6. Paragraph length (tokens) by detector class.
F-7. A human contribution ratio in the spirit of NIRVANA: share of final tokens whose class is T (typed in small edits), compared with 1 - U* and with T + E.

## RelianceScope (src/r9_rs.py)

K-1. Decomposition of the chance baseline under the three lexers: share of null-covered n-grams of the final code that are reference-solution n-grams, to explain why bound (a) and chance-corrected uptake diverge under the code lexers.
K-2. Runtime per student.

## Audit of sampled passages against evidence the attribution does not use (src/r9_flora.py, AUDIT)

A-1. For the 71 FLoRA students whose editor and chat-box keystrokes and chatbot-panel text selections are logged, maximal runs of at least eight tokens of one class (C, E, T, X) are sampled at random (seed 2091; at most three per student; up to 50 per class). The attribution uses dialogue, snapshots and shared sources only; the audit uses keystrokes and selections. Criteria, fixed before the audit was run:
- T agrees if the printable editor keystrokes in the entry interval of the passage (from 3 s before its first n-gram entered to 1 s after its last) number at least half its characters and no Control/Meta+V occurred.
- X and C agree if those keystrokes number fewer than a fifth of its characters (the text was not typed in the editor). For C we also report the share selected in the chatbot panel before entry, evidence of chatbot origin.
- E agrees if there is student-side evidence before the chatbot repeated the text: for prompt-first E, chat-box keystrokes numbering at least half the passage characters, or a Control/Meta+V in the chat box, in the 10 minutes before the ask; for editor-first E, the passage entered the editor before the first reply that contains it, with editor keystrokes or a paste shortcut at entry. We also report the share of E passages selected in the chatbot panel before entry, which would contradict the label.
Per-class agreement with intervals from 2,000 resamples of students. The sample is released (results/r9_audit_sample.csv) for manual replay coding.
