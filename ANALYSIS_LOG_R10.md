# Analysis log entry R10

Written after all earlier results (S1 to R9) were known, and committed before the R10 analyses ran. All analyses below are exploratory. None changes a measure, a class definition or a parameter value of the primary attribution (n = 8; tau = 5 for FLoRA and 20 for RelianceScope; delta = 0; peer threshold 5; token priority C > E > S > T > X; alphanumeric tokens).

## B. Benchmark with planted provenance, fitted to FLoRA (src/r10_benchmark.py)

B-1. Simulated students are built from the FLoRA logs. Each simulated student borrows the chatbot replies, prompt lengths and editor timing of one real FLoRA student (sampled with replacement) and writes a document from blocks with a known route. Block lengths, the gap between snapshots during typing, and the gap between blocks are drawn from the empirical FLoRA distributions (edit sizes of insertions above tau, inter-snapshot intervals below 60 s, intervals between large insertions). Text is real: chatbot text comes from the borrowed student's replies, student text from other students' final proposals, shared text from reading highlights, and outside-AI text from replies given to other students.
Routes and true class:
- C-paste: a reply passage pasted verbatim (C).
- C-retype: a reply passage retyped token by token (C).
- C-para-p: a reply passage pasted after a share p of its tokens was replaced (p = 0.1, 0.25), (C).
- E-prompt: the student's own draft sent in a prompt, repeated in the reply and pasted back from the reply (E).
- E-editor: text typed in the editor and later repeated by the chatbot (E).
- E-deleted: text typed, deleted, sent in a prompt and pasted back from the reply (E).
- S: a reading highlight pasted (S).
- X-ai: text from an AI tool outside the environment (a reply given to another student) pasted (X).
- X-other: text from another source outside the logs pasted (X).
- M: a typed block cut and pasted elsewhere in the document (T: the text was composed in the editor).
- T: genuine composition typed in small edits (T).
B-2. Outcomes: per-class confusion matrix, precision, recall and F1 at token level; student-level error of U* against the true C share (mean absolute error, mean signed error, Spearman correlation, agreement on "at least half from the chatbot"). Conditions: full snapshot resolution and snapshots thinned to at least 5, 15, 60 and 300 s apart. 400 students per condition, seed 2100.
B-3. Baselines on the same simulated logs, scored for the binary decision "chatbot text" (true class C) at token level and for the student-level share: (a) naive overlap, any token covered by an 8-gram of a reply; (b) direction-blind removal, (a) minus 8-grams that occur in any prompt; (c) dialogue only, reply 8-grams not in an earlier prompt; (d) winnowing fingerprints in the manner of MOSS (Schleimer et al., 2003), 8-token k-grams hashed, window w = 4, a final-text fingerprint that is also a reply fingerprint marks its k-gram; (e) greedy string tiling in the manner of JPlag (Prechelt et al., 2002), minimum match length 8 tokens, between the final text and the concatenated replies; (f) the online version of the attribution (O-1). For the typed share: the true T share against the estimated T share (a human contribution ratio in the spirit of NIRVANA) and against 1 minus each baseline's chatbot share.

## O. Online attribution, run time and memory (demo/provenance_online.py, src/r10_online.py)

O-1. An incremental version of Algorithms 1 and 2 processes the events of one student in time order (reply, prompt, snapshot). At each snapshot it labels the n-grams that enter the editor at that moment from the events seen so far (C, E prompt-first, S, T or X); n-grams that enter before a reply contains them are relabelled E when that reply arrives (editor-first echo). Recurrence across the cohort's proposals is not known while students write, so online S uses reading highlights only.
O-2. On FLoRA and RelianceScope: agreement of the online n-gram labels at submission with the offline labels (share of tokens with the same class, Cohen's kappa), share of tokens whose online label at entry changed later, latency per snapshot (median, 99th percentile, maximum), and peak memory per student (tracemalloc) for the online and the offline versions, with total time per cohort on one CPU core. Machine described in the results file.

## W. Winnowing, string tiling and contribution ratio on the study data (src/r10_baselines.py)

W-1. FLoRA: for every student, the share of the final proposal marked by (d) winnowing and (e) string tiling against the student's own replies; median, share of students at or above 50%, Spearman correlation with U*, and the share of marked tokens that fall in each of the five classes. Chance level: the same measures against the replies of other students (median over 20 random other students).
W-2. RelianceScope: the same student-level measures for the final code, with and without the template, and against other students' replies; segment-level agreement with the human codes "copy the response" and "write over the response" (one against rest, AUC with student-cluster bootstrap intervals) for winnowing and tiling of each segment's replies against the final code, compared with segment uptake (Algorithm 4).
W-3. A human contribution ratio (share of class T) for RelianceScope, as reported for FLoRA in R9 F-7.

## K. RelianceScope (src/r10_rs.py)

K-1. Attribution of intermediate snapshots: for each coded segment, the artefact is the last editor snapshot before the next segment starts (instead of the final code); segment uptake counts chatbot-first n-grams of the segment's replies present in that snapshot and entered during the segment. AUC for "copy the response" and "write over the response" at n = 5 and 8, alphanumeric and code lexer; the same with the snapshot 2 and 5 minutes after the segment's last reply.
K-2. Effect of each parameter choice on the RelianceScope quantities, as for FLoRA: n (5, 8, 12), tau (5, 20, 50), delta (0, +60 s), token projection (C first, E before C, majority), lexer (alphanumeric, code, code with identifiers normalised). Outcomes: mean and median U*, chance-corrected mean, share at or above 50%, AUC for "copy the response", post-test slope per SD of U* (OLS on pre-test, HC3).
K-3. Label-noise ceiling for the copy code over q = 0, 0.02, 0.04, ..., 0.20 (simulated AUC of a perfect score, 10,000 draws).

## P. FLoRA paste checks (src/r10_flora_checks.py)

P-1. Paste-C, and the shares of T and X 8-grams entering with a logged paste, with 95% intervals from 2,000 resamples of the 68 students with logged pastes.
P-2. Completeness of the editor log: snapshots per 100 final tokens, median interval between snapshots during writing sessions, and the share of large insertions (> tau tokens), for students with and without logged pastes.
P-3. Clock differences near 1, 10 and 11 h between asks and chat-box Enter keystrokes: number of asks and students, and U* for those students recomputed with delta = +1 h and with the affected asks shifted by the nearest whole hour.
