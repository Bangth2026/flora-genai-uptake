# Analysis log entry R24

Written on 22 November 2026, after a review of the eighth version of the manuscript and after the NIRVANA project on OSF (the hold-out set of R22) stopped being publicly accessible. NIRVANA is replaced as the hold-out set by CoAuthor (Lee, Liang & Yang, 2022), a public dataset whose platform logged every accepted model suggestion as an event of its own. This entry and `src/r24_coauthor.py` are committed and pushed before the script is run on the full dataset; the results are added in a later commit of the same entry.

## What was done before this entry was written

CoAuthor was downloaded from the link on https://coauthor.stanford.edu (Google Drive file 1C9FCCsyY-5I7mcBHi-__R7lxHkGX_-9Q; coauthor-v1.0.zip, 49,956,179 bytes, md5 f5d815a0243bdbced41ffddea3ea503c; 1,447 files, one JSON-lines file per session) at 10:42 UTC. To write the readers we looked at: the event names, event sources and field names counted over the first 50 files; and the first event of each (name, source) pair in one file. A dry run on three sessions checked only that the script executes; no label, share, U or comparison with the acceptance signal was looked at before this entry was pushed.

## A. Hold-out application (src/r24_coauthor.py)

The attribution is executed from the study code, not re-written: `scan`, `classify`, `project` and `propagate` are read from `src/r9b_flora_token_entry.py` and run as they are, with n = 8, tau = 5, delta = 0, the peer threshold of five other documents, the token priority C > E > S > T > X and the token entry rule. Readers (new): snapshots after every text-insert and text-delete event, rebuilt from the initial document and the Quill deltas; replies R = every suggestion shown at a suggestion-open event ('original' and 'trimmed' text); writer turns Q = the document before the cursor at each suggestion-get event, the context sent to the model; shared sources H = the 8-grams of each session's initial prompt plus the peer rule over the final documents; the document A = the final document without the initial prompt when it still begins with it.

Independent signal: a token of A is suggestion-inserted when the transition that last inserted it is a text-insert event with eventSource 'api', which the platform writes when the writer accepts a suggestion. The signal reads no text match. It misses suggestion text that the writer retyped.

Pre-specified outputs, reported whatever they show:

- A-1. Share of C tokens that were suggestion-inserted, and share that entered by typing.
- A-2. Distribution over C, E, S, T, X of the suggestion-inserted tokens (recall of C against the signal).
- A-3. Share of E, S, T and X tokens that were suggestion-inserted.
- A-4. Session level: Spearman correlation of U with the suggestion-inserted share, with a 95% interval from 2,000 resamples of sessions (seed 2401); the same for naive 8-gram overlap and the paired difference; mean absolute differences; shares at or above half and agreement at half.
- A-5. Naive 8-gram overlap scored on the signal: share of suggestion-inserted tokens it marks and share of the tokens it marks that were suggestion-inserted.
- A-6. Agreement by stratum: suggestion-inserted tokens and all other tokens, with the entry route of C tokens in the second stratum.
- A-7. Echo: E tokens, how many were suggestion-inserted, how many naive overlap marks.
- A-8. Coincidence (case iii of Property 1): share of C without a suggestion event; typed C runs.
- A-9. Sizes of the writer's insert events (share above tau; 99th percentile by event and by token) and the entry route of X.

Sensitivity, pre-specified: H without the initial prompts (`noprompt_H`); replies from the 'trimmed' suggestion text only (`trimmed_only`); tau set to the 99th percentile by event from A-9 (`tauK`). There is no threshold for matching the signal, because the platform logs the acceptance itself.

No threshold for success is set. The parameters are those fixed on FLoRA and are not tuned on CoAuthor. CoAuthor differs from FLoRA in model (GPT-3), interaction (inline continuation suggestions on request, not a chat), writers (63 crowd workers, 1,445 sessions; sessions are resampled, not writers, because the release does not identify writers) and task (short creative and argumentative pieces in one session).

## Status of NIRVANA

The analyses of R22 and R23 on NIRVANA stay in the repository and its history, but the article no longer uses them.
