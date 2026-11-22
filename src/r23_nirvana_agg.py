"""Analysis log R23, N-10 to N-12 (NIRVANA), computed from the stored outputs of R22 (results/r22_nirvana.json), because
the NIRVANA project on OSF was no longer publicly accessible when this entry was written and the raw data could not be
downloaded again. Nothing is re-run on the raw data and no label changes.
N-10 agreement stratified by how a token entered the editor: tokens that entered by a paste matched to a copy from a
ChatGPT reply (the stratum in which the label and the copy signal see the same verbatim paste) and all other tokens;
the 2 x 2 of copy event x C in each stratum.
N-11 naive 8-gram overlap on the same copy signal: share of copied tokens it marks (a lower bound: copied tokens labelled
C or E, which naive overlap marks by definition), share of marked tokens that were copied, and student-level Spearman
correlation with the dataset's own share of pasted ChatGPT text, with the paired bootstrap difference against U
(2,000 resamples of students, seed 2302). The cells where the two methods differ are the echo cells.
N-12 what the copy signal does and does not bound for coincidence (case iii of Property 1): a C token that entered by a
paste of text copied from a reply came from that reply, so coincidence is at most the share of C tokens without such a
copy event; the 5.3% figure of the eighth version is that share restricted by two assumptions, stated here.
Usage: python -I src/r23_nirvana_agg.py -> results/r23_nirvana_agg.json"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results')
d = json.load(open(os.path.join(R, 'r22_nirvana.json'))); ph = json.load(open(os.path.join(R, 'r22_nirvana_posthoc.json')))
cb = pd.DataFrame(d['class_by_entry_counts']).fillna(0).astype(int)        # rows: entry route, columns: class
out = {'source': 'results/r22_nirvana.json and results/r22_nirvana_posthoc.json (R22); raw data not re-read'}
# ---------- N-10
gpt = cb.loc['paste:gpt']; rest = cb.drop(index='paste:gpt').sum()
tot = int(cb.values.sum()); C = int(cb['C'].sum())
out['N10_stratified'] = {
    'tokens': tot,
    'verbatim_chatgpt_paste': {'tokens': int(gpt.sum()), 'share_of_tokens': float(gpt.sum() / tot), 'C': int(gpt['C']), 'not_C': int(gpt.sum() - gpt['C']),
                               'share_labelled_C': float(gpt['C'] / gpt.sum()), 'by_class': {k: int(v) for k, v in gpt.items()}},
    'all_other_tokens': {'tokens': int(rest.sum()), 'share_of_tokens': float(rest.sum() / tot), 'C': int(rest['C']), 'not_C': int(rest.sum() - rest['C']),
                         'share_labelled_C': float(rest['C'] / rest.sum()), 'by_class': {k: int(v) for k, v in rest.items()},
                         'C_by_entry': {k: int(v) for k, v in cb['C'].drop(index='paste:gpt').items()}},
    'share_of_C_in_verbatim_stratum': float(gpt['C'] / C),
    'note': 'every token outside the verbatim stratum has no copy event, so the copy signal cannot confirm or refute a C label there'}
# ---------- N-11
ps = pd.DataFrame(d['per_student'])
naive_tok = float((ps.naive * ps.tokens).sum()); copied_CE = int(gpt['C'] + gpt['E'])
rng = np.random.default_rng(2302); diff, rU, rN = [], [], []
for _ in range(2000):
    i = rng.integers(0, len(ps), len(ps)); x = ps.iloc[i]
    a = spearmanr(x.U, x.gpt_final_pct)[0]; b = spearmanr(x.naive, x.gpt_final_pct)[0]; rU.append(a); rN.append(b); diff.append(a - b)
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out['N11_naive_on_copy_signal'] = {
    'copied_tokens': int(gpt.sum()),
    'copied_tokens_marked_by_naive_at_least': float(copied_CE / gpt.sum()), 'copied_tokens_labelled_C': float(gpt['C'] / gpt.sum()),
    'naive_marked_tokens': naive_tok, 'naive_marked_tokens_copied_at_most': float(copied_CE / naive_tok) if naive_tok else None,
    'C_tokens_copied': float(gpt['C'] / C),
    'spearman_with_gpt_final_pct': {'U': [float(spearmanr(ps.U, ps.gpt_final_pct)[0])] + q(rU), 'naive': [float(spearmanr(ps.naive, ps.gpt_final_pct)[0])] + q(rN),
                                    'U_minus_naive': [float(np.mean(diff))] + q(diff)},
    'mean_abs_difference_with_gpt_final_pct': {'U': float((ps.U - ps.gpt_final_pct).abs().mean()), 'naive': float((ps.naive - ps.gpt_final_pct).abs().mean())},
    'agreement_at_half_with_gpt_final_pct': {'U': float(((ps.U >= .5) == (ps.gpt_final_pct >= .5)).mean()), 'naive': float(((ps.naive >= .5) == (ps.gpt_final_pct >= .5)).mean())},
    'share_at_or_above_half': {'U': float((ps.U >= .5).mean()), 'naive': float((ps.naive >= .5).mean()), 'gpt_final_pct': float((ps.gpt_final_pct >= .5).mean())},
    'students_where_naive_differs_from_U_by_more_than_half_a_point': int(((ps.naive - ps.U).abs() > .005).sum()),
    'echo_cells': {'E_tokens': int(cb['E'].sum()), 'E_copied': int(gpt['E']), 'E_not_copied': int(cb['E'].sum() - gpt['E']),
                   'share_of_E_copied': float(gpt['E'] / cb['E'].sum())}}
# ---------- N-12
noC = C - int(gpt['C']); tr = ph['N7_typed_C_runs']
out['N12_coincidence'] = {'C_tokens': C, 'C_tokens_without_chatgpt_copy_event': noC, 'upper_bound_share_of_C': float(noC / C),
    'typed_C_tokens': int(cb.loc['typed', 'C']), 'typed_C_in_runs_lt_12_share_of_C': float(tr['share_tokens_in_runs_lt_12'] * tr['tokens'] / C),
    'typed_C_in_runs_ge_16_share_of_C': float(tr['share_tokens_in_runs_ge_16'] * tr['tokens'] / C),
    'C_in_pastes_without_copy_event': int(cb.loc['paste:unmatched', 'C']), 'C_other_routes': int(noC - cb.loc['typed', 'C'] - cb.loc['paste:unmatched', 'C']),
    'reading': 'coincidence is at most 19.1% of C; it is at most 5.3% only if typed runs of 12 or more tokens are retyped reply passages and C text pasted without a copy event came from a reply, neither of which the stored outputs can test'}
json.dump(out, open(os.path.join(R, 'r23_nirvana_agg.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
