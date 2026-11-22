# Analysis log entry R22

Written on 22 November 2026, in response to a review of the seventh version of the manuscript that asked for the frozen method to be applied, unchanged, to a second public dataset with an independent copy signal. This entry and `src/r22_nirvana.py` are committed and pushed before the script is run on the full dataset; the results are added in a later commit of the same entry.

## What was done before this entry was written

NIRVANA (Jelson et al., 2026) was downloaded from the OSF project 3a8uh (file `dataset.xlsx`, 2,863,378 bytes, md5 cf7d122ddf299b6885f83c200a3ef035) at 09:24 UTC. To write the readers we looked at: the sheet and column names; the number of operations by pane and type; the first rows of each operation type; whether the last editor snapshot has the same tokens as the column `Final Text` (61 of 77; in the other 16 `Final Text` holds a list representation whose escaped tab characters add a token `t`); and whether editor pastes can be matched to an earlier copy event by their first 20 tokens (130 to a ChatGPT reply, 9 to the editor, 2 to the essay prompt, 32 not matched by that crude rule). A dry run of the script on three students checked only that it executes; no class label, U, share or comparison with the copy signal was looked at before this entry was pushed.

## N. Hold-out application (src/r22_nirvana.py)

The attribution is executed from the study code, not re-written: the functions `scan`, `classify`, `project` and `propagate` are read from `src/r9b_flora_token_entry.py` and executed as they are, with n = 8, τ = 5, δ = 0, the peer threshold of five other documents, the token priority C > E > S > T > X and the token entry rule. Only the readers are new (replies: `gpt_response`; prompts: `gpt_inquiry`; snapshots: `current_editor` after every editor operation; final text: the last snapshot; shared texts H: every text copied from the essay-prompt pane, plus the peer rule over the 77 final texts).

Copy signal, independent of the rules: each editor paste is matched to the latest earlier copy or cut event in any pane whose tokens cover at least half of the tokens the paste inserted; the paste's source is that pane (`gpt` = a ChatGPT reply) or `unmatched`. A token of the final text is ChatGPT-copied if the transition that last inserted it is a paste from `gpt`.

Pre-specified outputs, reported whatever they show:

- N-1. Share of C tokens that entered by a ChatGPT-copied paste, and share that entered by typing (a C token can be retyped chatbot text, which the copy signal does not see, so the first share is a lower bound on precision).
- N-2. Distribution over C, E, S, T, X of the ChatGPT-copied tokens (recall of C against the copy signal).
- N-3. Share of E, T and X tokens that are ChatGPT-copied.
- N-4. Student level: Spearman correlation of U with the dataset authors' own `GPT Final %` (share of the final essay from pasted ChatGPT text) and with the share of ChatGPT-copied tokens; mean absolute difference; share of students at or above half by each measure and agreement at half.
- N-5. Naive overlap (any 8-gram of a reply) for comparison, as in Section 5.3 of the article.

No threshold for success is set. The parameters are those fixed on FLoRA and are not tuned on NIRVANA. NIRVANA differs from FLoRA in model (GPT-3.5-turbo), task (a short analytical essay in one session), students and editor (snapshots after every operation, with paste and copy events).

## Results of the pre-specified outputs (run at 09:27 UTC, after the plan above was pushed in commit 9c02c2b)

77 students, 29,692 tokens of final text; 6 students sent no prompt. Mean class shares: C 47.6%, E 5.0%, S 0.6%, T 24.2%, X 22.6%.

- N-1. Of the C tokens, 80.9% entered by a paste of text copied from a ChatGPT reply and 9.5% by typing; the rest entered by pastes with no matching copy event (6.9%), by moves within the editor (0.9%) or by other operations (1.7%). No C token entered by a paste whose source was the essay prompt.
- N-2. Of the ChatGPT-copied tokens of the final texts, 96.2% were labelled C, 2.8% E, 0.8% X and 0.1% T.
- N-3. ChatGPT-copied: 18.7% of E tokens, 1.5% of X tokens, 0.2% of T tokens.
- N-4. Spearman correlation of U with the dataset's `GPT Final %` 0.88 and with the copied share 0.85; mean absolute difference 12.2 points; mean U 47.6% against 41.0%; at or above half: 50.6% of students by U, 41.6% by `GPT Final %`, 45.5% by the copied share; agreement at half 83%.
- N-5. Naive overlap: mean 52.6% of the text; 55.8% of students at or above half.

## Post hoc analyses (written after the results above were seen; src/r22_nirvana_posthoc.py, src/r22_flora_survey.py, src/r22_benchmark.py)

- N-6. Tolerant matching on NIRVANA (one token may differ): C 47.4%; 80.2% of C tokens ChatGPT-copied; 95.2% of copied tokens labelled C; ChatGPT-copied E 22.9%, X 1.0%.
- N-7. C tokens without a ChatGPT copy event. Typed C: 1,384 tokens in 301 runs from 28 students, median run 2 tokens; 55% of these tokens lie in runs shorter than 12 tokens (words typed inside or at the edge of pasted chatbot passages, which the token priority assigns to C) and 38% in runs of at least 16 tokens (retyped chatbot passages); one typed run consisted only of 8-grams that also occur in replies given to other students. Pastes without a matching copy event: 1,004 C tokens from 4 students.
- N-8. X tokens: 81.8% entered by insert operations of more than five tokens, 14.8% by pastes with no matching copy event, 1.5% by ChatGPT-copied pastes. 7.8% of the editor's insert operations insert more than five tokens (99th percentile of non-empty inserts: 13 tokens): the NIRVANA editor logs some typing in chunks, so assumption A2 holds only in part and the frozen tau = 5 sends chunked typing to X.
- N-9. With tau = 13 taken from NIRVANA's own logs: T 41.4%, X 5.5%; C, E, S and every comparison with the copy signal for C are unchanged.
- F-50. FLoRA post-task survey, free-text question on other tools: 11 of 180 respondents (6.1%) named an outside AI chatbot (ChatGPT, Claude, Copilot); their mean shares were C 25.9%, X 52.6%, X-with-C 23.9%, against C 20.6%, X 52.0%, X-with-C 20.2% for the other 166 respondents with labels.
- B-47. Tolerant matching on the set with model-revised chatbot text (R21 B-46): C precision 0.912 and recall 0.912, against 0.996 and 0.825 for the attribution; surviving reply text labelled C after a model rewrite 0.85 (attribution 0.64); the model's own words labelled C 31-46% (attribution 0.6-5.4%); echo routes 24-25% (attribution 0%).

The token-level NIRVANA labels are not released, because the NIRVANA files state no licence; the aggregate and per-student results are in `results/r22_nirvana.json` and `results/r22_nirvana_posthoc.json`, and the script recomputes the labels from the dataset.

## Correction to R21

The results of R21 B-46 report the share of true chatbot text labelled C after condensing as 0.67; `results/r21_benchmark.json` gives 0.6645, which rounds to 0.66. The seventh version of the manuscript carried the same rounding error; the eighth corrects it. No other number is affected.
