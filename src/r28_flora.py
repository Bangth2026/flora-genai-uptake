"""Analysis log R28, F-57 to F-61 (FLoRA). Written in response to a review of the thirteenth version of the manuscript,
after the analyses ran; no rule, parameter or label of the attribution changes. Definitions of
src/r9b_flora_token_entry.py (n = 8, tau = 5, delta = 0, peer threshold 5), executed through src/r23_flora.py up to F-51.
F-57 the subgroups with process evidence against the rest of the cohort: writers with editor keystrokes (the audit and
census subgroup), writers with logged pastes, and either; mean shares of C, E, T, X and X-with-C, the share of writers at
or above half chatbot-first text, and subgroup-minus-rest differences with student-bootstrap intervals (2,000 resamples,
seed 2801).
F-58 the writers at or above half chatbot-first text: how many have keystrokes or logged pastes, and, for those with
keystrokes and chatbot-first runs of at least eight tokens, the share of runs selected in the chatbot panel before entry.
F-59 a strict lower end: chatbot-first tokens that entered the editor in an insertion of more than tau tokens (C tokens
that entered by typing-sized insertions, i.e. words typed inside or at the edge of a chatbot-first passage, are left out).
F-60 the identified-range ladder with tau = 2 (the data-driven value of R23 F-52) in place of tau = 5: C, E and S do not
depend on tau; T and X are recomputed; X-with-C is kept at its tau = 5 value (see the note in the output).
F-61 the token priority: share at or above half with C before E (as reported) and with E before C (projection of R23).
Usage: python -I src/r28_flora.py DATA_DIR [TAG] -> results/r28_flora[_TAG].json and results/r28_flora_students[_TAG].csv
(run with DATA_DIR holding the whole trace as trace.pkl; TAG firstsheets: the same with the first sheets only, as read by
every earlier entry, kept for the subgroup comparison F-57)"""
import sys, os, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1]; TAG = ('_' + sys.argv[2]) if len(sys.argv) > 2 else ''
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)
src23 = open(os.path.join(HERE, 'r23_flora.py')).read()
cut23 = src23.index('# ---------- F-51 partial identification')
G = {'__name__': 'r23_defs', '__file__': os.path.join(HERE, 'r23_flora.py')}
sys.argv = [os.path.join(HERE, 'r23_flora.py'), D]
exec(compile(src23[:cut23], 'r23_flora.py', 'exec'), G)
TK, users, KEYU, grams, tok = (G[k] for k in ('TK', 'users', 'KEYU', 'grams', 'tok'))
N = 8; NB = 2000; rng = np.random.default_rng(2801)
share = lambda m: float(m.mean()) if len(m) else 0.0
out = {}

# per-writer shares (as R23 F-51)
XW = pd.read_csv(os.path.join(HERE, '..', 'results', 'r23_flora_xwithc.csv')).set_index('user_id').X_with_C
rows = []
for u in users:
    t = TK[u]; lab = t['lab']; F = t['F']; te = t['te']
    after = np.zeros(len(F), bool)
    for i, g in enumerate(t['FG']):
        if t['cls'][g] == 'E' and g in t['tbot']:
            te_ = t['ted'].get(t['FGs'][g])
            if te_ is not None and pd.Timestamp(te_) >= pd.Timestamp(t['tbot'][g]): after[i:i + N] = True
    E = lab == 'E'
    notCES = np.isin(lab, ['C', 'E', 'S'], invert=True)
    rows.append(dict(u=u, n_tokens=len(F), C=share(lab == 'C'), C_strict=share((lab == 'C') & (te == 2)), C_typed_sized=share((lab == 'C') & (te == 1)),
                     E=share(E), E_after=share(E & after), S=share(lab == 'S'), T=share(t['var'] == 'T'), X=share(t['var'] == 'X'),
                     T_tau2=share(notCES & (t['TEV'][2] == 1)), X_tau2=share(notCES & (t['TEV'][2] == 2)),
                     X_with_C=float(XW.get(u, 0.0))))
S = pd.DataFrame(rows).set_index('u')
log('shares', len(S), S.C.mean())

# process-evidence subgroups (as R23)
tr = pd.read_pickle(f'{D}/trace.pkl')
PASTEU = set(tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].user_id.unique())
SELU = set(tr[(tr.learning_action_label == 'READ_FEEDBACK_MULTI_AGENTS') & tr.event_value.astype(str).str.startswith('MOUSE_SELECT:::')].user_id.unique())
del tr
S['keystrokes'] = S.index.isin(KEYU); S['pastes'] = S.index.isin(PASTEU); S['either'] = S.keystrokes | S.pastes
S['panel_selections'] = S.index.isin(SELU)
S['half'] = (S.C >= .5).astype(float)

def boot_diff(col, grp):
    a = S[col].values; g = S[grp].values; n = len(a); d = []
    for _ in range(NB):
        i = rng.integers(0, n, n); aa, gg = a[i], g[i]
        if gg.sum() and (~gg).sum(): d.append(aa[gg].mean() - aa[~gg].mean())
    return [float(a[g].mean() - a[~g].mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
F57 = {}
for grp in ('keystrokes', 'pastes', 'either'):
    g = S[grp].values
    F57[grp] = {'n_in': int(g.sum()), 'n_rest': int((~g).sum()),
                'mean_in': {k: float(S.loc[g, k].mean()) for k in ('C', 'E', 'T', 'X', 'X_with_C', 'half')},
                'mean_rest': {k: float(S.loc[~g, k].mean()) for k in ('C', 'E', 'T', 'X', 'X_with_C', 'half')},
                'diff_in_minus_rest': {k: boot_diff(k, grp) for k in ('C', 'E', 'X', 'half')},
                'tokens_median_in_rest': [float(S.loc[g, 'n_tokens'].median()), float(S.loc[~g, 'n_tokens'].median())]}
F57['overlap'] = {'keystrokes_and_pastes': int((S.keystrokes & S.pastes).sum()), 'keystrokes_only': int((S.keystrokes & ~S.pastes).sum()),
                  'pastes_only': int((~S.keystrokes & S.pastes).sum()), 'neither': int((~S.either).sum()),
                  'with_panel_selection_events': int(S.panel_selections.sum())}
out['F57_subgroups'] = F57
log('F57', {k: (v['n_in'], round(v['mean_in']['half'], 3), round(v['mean_rest']['half'], 3)) for k, v in F57.items() if k != 'overlap'})

# F-58 writers at or above half
H = S[S.C >= .5]
r23 = json.load(open(os.path.join(HERE, '..', 'results', 'r23_flora.json')))
# census runs per writer are recomputed from the labels and the selection events, as in R23 F-53
sel_src = src23[src23.index('# ---------- trace for F-53 and F-54'):src23.index('# ---------- F-53 census')]
exec(compile(sel_src, 'r23_flora.py', 'exec'), G)
cr = []
for u in KEYU:
    for c, i, j in G['runs_of'](TK[u]['lab'], 'C'):
        ev = G['evidence'](u, i, j)
        if ev: cr.append(dict(u=u, tokens=j - i, selected=ev['selected']))
cr = pd.DataFrame(cr)
per = cr.groupby('u').apply(lambda x: (x.selected * x.tokens).sum() / x.tokens.sum(), include_groups=False)
hk = [u for u in H.index if u in per.index]
out['F58_writers_at_or_above_half'] = {
    'writers': int(len(H)), 'with_keystrokes': int(H.keystrokes.sum()), 'with_pastes': int(H.pastes.sum()), 'with_either': int(H.either.sum()),
    'with_keystrokes_and_C_runs': len(hk),
    'of_those_selected_token_share_ge_80': int((per.loc[hk] >= .8).sum()) if hk else 0,
    'of_those_selected_token_share_median': float(per.loc[hk].median()) if hk else None,
    'census_check_runs': int(len(cr)), 'census_check_selected_share': float(cr.selected.mean())}
log('F58', out['F58_writers_at_or_above_half'])

# F-59 strict lower end
def at(v, thr_):
    v = np.asarray(v); b = [np.mean(v[rng.integers(0, len(v), len(v))] >= thr_) for _ in range(NB)]
    return [float(np.mean(v >= thr_)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out['F59_strict_lower_end'] = {'mean_C': float(S.C.mean()), 'mean_C_strict': float(S.C_strict.mean()), 'mean_C_typed_sized': float(S.C_typed_sized.mean()),
                               'share_of_C_tokens_typed_sized': float((S.C_typed_sized * S.n_tokens).sum() / (S.C * S.n_tokens).sum()),
                               **{f'at_or_above_{int(q * 100)}': {'C': at(S.C, q), 'C_strict': at(S.C_strict, q)} for q in (.25, .5, .75)}}
log('F59', out['F59_strict_lower_end']['at_or_above_50'])

# F-60 ladder with tau = 2
lad = [('L0 C', S.C), ('L1 + E-after', S.C + S.E_after), ('L2 + E-after + X-with-C', S.C + S.E_after + S.X_with_C),
       ('L3 + E-after + all X (tau 5)', S.C + S.E_after + S.X), ('L3 + E-after + all X (tau 2)', S.C + S.E_after + S.X_tau2),
       ('L4 + all E + all X (tau 5)', S.C + S.E + S.X), ('L4 + all E + all X (tau 2)', S.C + S.E + S.X_tau2)]
out['F60_tau2'] = {'mean_T_tau5': float(S['T'].mean()), 'mean_T_tau2': float(S.T_tau2.mean()), 'mean_X_tau5': float(S.X.mean()), 'mean_X_tau2': float(S.X_tau2.mean()),
                   'ladder': [{'step': n_, 'mean': float(v.mean()), 'at_or_above_50': at(v, .5)} for n_, v in lad],
                   'note': 'X-with-C is kept at its tau = 5 value: an insertion of 3 to 5 tokens that becomes X under tau = 2 contains no complete 8-gram of its own, so it can join X-with-C only by sharing an inserted segment with chatbot-first tokens, which a segment of at most 5 tokens cannot do unless an 8-gram spanning old and new text is C'}
log('F60', [(r['step'], r['at_or_above_50'][0]) for r in out['F60_tau2']['ladder']])

# F-61 token priority
F51 = r23['F51_partial_identification']
out['F61_token_priority'] = {'C_before_E_at_or_above_50': at(S.C, .5)[0], 'E_before_C_projection_reported_in_R23': 0.16,
                             'note': 'the E-before-C projection is the n-gram-to-token projection of R23 (Table 4: 16.0%)'}
out['seconds'] = round(time.perf_counter() - t00)
S.to_csv(os.path.join(HERE, '..', 'results', f'r28_flora_students{TAG}.csv'), float_format='%.6f')
json.dump(out, open(os.path.join(HERE, '..', 'results', f'r28_flora{TAG}.json'), 'w'), indent=1, default=float)
log('done')
