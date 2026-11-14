# Analysis log entry R13

Written after all earlier results (S1 to R12) were known and, like R12, after its analyses had run. All analyses are exploratory. No measure, class definition or parameter value changes, and no result with delta = 0 changes; the primary attribution, U\* and all class shares of R11 and R12 are unchanged.

## Decision list applied as written when delta > 0

D-3. Offline R2 (`demo/provenance.py`; `src/r13_benchmark.py` applies the same change to the definitions of `src/r11_benchmark.py` that it executes). R2 (editor-first echo) requires t_W(g) + delta < t_R(g). The code of R11 and R12 tested t_W(g) < t_R(g). The two differ only for a shared n-gram whose first reply came within delta after it entered the editor: R2 as written does not hold, so R3 gives S, whereas the code gave E. With delta = 0 the conditions are identical.

D-4. Online R1 for S labels (`demo/provenance_online.py`). R1 precedes R3. When a reply arrives within delta after an n-gram entered the editor with label S and a writer turn contained the n-gram before that reply, R1 holds offline and the label is E. The R12 online version changed only T and X labels in this window and left S unchanged; it now turns S into E in this case. Otherwise S stays S (R3 precedes R4) and T or X becomes C (R4) or E (R1), as before. With delta = 0 the window is empty.

## B. Benchmark (src/r13_benchmark.py; same simulated students as R11 and R12)

B-13. Online against offline labels at delta = 30 s (13 and 20 routes) and delta = 0 (20 routes), for both online versions (R12, R13) against both offline versions (R12, R13), with every n-gram whose labels differ described by its offline and online label, whether it is shared, whether a writer turn preceded the reply and whether the first reply came within delta after entry.

Result: with delta = 30 s on the 20 routes, the R12 offline and online labels differed on 3 tokens (0.002%; 10 n-grams, all shared, first reply within delta after entry, offline E, online S: reading highlights inside text copied from streaming replies). With R2 as written (D-3) offline and online labels agree on every token, for both online versions; no S label met R1 within delta, so D-4 changes nothing on the benchmark. With delta = 0 all versions agree on every token.

B-14. Offline shares by route with delta = 30 s and R2 as written (20 routes): streamed text 97.8% C, editor-first echo passed by the environment 7.8% C; chatbot-first precision 0.995, recall 0.793. These equal the R11 values to the precision reported.

B-15. Sources of the remaining AURC difference between the attribution and earliest plausible attribution with abstention by entry size (decision "chatbot text", scores of R12): for each route, its tokens, the tokens each method decides and gets wrong, and the AURC difference with the route's tokens left out.

Result, 20 routes: difference -0.0053; without the shared-source route -0.0037 (R3 decided 3,136 of its 3,148 tokens, all correctly; earliest plausible attribution with abstention left all of them undecided); without text copied from streaming replies -0.0034 (both methods wrong on the same 3,372 tokens; they differ in ranking); every other route left out -0.0044 to -0.0067. 13 routes: difference -0.0015; without the shared-source route -0.0005 (3,962 of 3,971 decided by the attribution, 108 by the other method); every other route left out -0.0007 to -0.0022.

## F. FLoRA

F-12 (`src/r13_flora.py`). Chatbot-panel selection before entry for the passages audited under the token entry rule (R11, F-5; same sample, random_state 2091; the keystroke criterion reproduces 0.96 for T and 0.94 for X): X 0 of 50 passages (32 students), T 3 of 50 (22 students).

F-14 (`src/r13_flora.py`). Unresolved text that entered with the first editor snapshot: 37.2% of all X tokens (38% of those present in the last snapshot, R12), 126 students. First snapshot length over all students: quartiles 1, 59 and 637 tokens; for the 126 students, 231, 642 and 737 tokens. 26 students' first snapshots share any 8-gram with the first snapshots of at least five others; of the 33,486 8-grams lying wholly in first-snapshot X text, 56 occur in another student's first snapshot and 118 (0.35%) in one to four other proposals (below the peer threshold), none in five or more. For 121 of the 126 students the first snapshot came after their first chatbot reply (5 students, 4.0% of these tokens, before). The editor did not load a common text; most of this X is a draft placed in the editor at once.

F-13 (`src/r13_wiki.py`). Overlap of each class with English Wikipedia (wikimedia/wikipedia, dump 20231101.en, 41 Parquet shards, 6,407,814 articles, read from the Hugging Face hub, each shard deleted after scanning). Every 8-gram lying wholly within one class of a proposal (156,628 distinct) was searched (Aho-Corasick) in the article text tokenised as in the attribution; 269 occur. Share of class tokens covered by such an 8-gram, pooled: C 0.91%, E 0.85%, S 36.9%, T 0.27%, X 0.84%; median over students 0 for every class; mean share of the proposal: C 0.23%, X 0.42%. Wikipedia is one public source among many; the result bounds the part of C that is encyclopaedic text the chatbot repeated, not web text in general.
