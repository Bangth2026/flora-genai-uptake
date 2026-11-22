"""Analysis log R23, F-51 to F-54 (FLoRA). Written in response to a review of the eighth version of the manuscript; no
rule, parameter or label of the attribution changes. Definitions of src/r9b_flora_token_entry.py (n = 8, tau = 5,
delta = 0, peer threshold 5), executed up to its audit.
F-51 partial identification (X-with-C from src/r23_flora_xsub.py, the partition of Table 5) of the share of writers whose text came mostly from the embedded chatbot: for each writer,
the shares of the token categories C, echo that entered the editor only after a reply containing it (E-after), other
echo (E-before), X-with-C, other X, T and S; then the share of writers at or above 25/50/75% under a nested ladder of
assumptions, each step adding to the chatbot count the categories that the dropped assumption no longer excludes
(student-bootstrap intervals, 2,000 resamples, seed 2301).
F-52 a data-driven rule for tau: the snapshot transitions of the keystroke students, each with its inserted tokens and the
printable editor keystrokes in the window of the R9 audit; typed = keystrokes for at least half of the inserted
characters. tau_hat = the 99th percentile of the sizes of typed insertions, weighted by inserted tokens (and, for
comparison, by insertions); the T and X shares and the share at or above half for tau_hat and a grid of tau.
F-53 census of every chatbot-first run of at least eight tokens of the keystroke students (the R9 audit sampled 50):
keystroke support, paste shortcut and selection in the chatbot panel before entry, with student-bootstrap intervals.
F-54 the seven chatbot-first passages of the R9 audit sample that were not selected in the chatbot panel: keystrokes,
paste shortcut, logged paste carrying the passage, and whether the passage's reply was opened in the separate
chatbot window, whose selections the trace may not record.
Usage: python -I src/r23_flora.py DATA_DIR -> results/r23_flora.json"""
import sys, os, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1]
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)
TAUS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 20)
src = open(os.path.join(HERE, 'r9b_flora_token_entry.py')).read()
src = src.replace("    TK[u] = dict(F=F, lab=lab, var=var, tt=tt, times=times)",
"""    TEV = {}
    for tau_ in TAUS:
        if tau_ == 5: TEV[tau_] = te; continue
        l2, tl2, _ = propagate(enc, tau_) if enc else (np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.int64))
        te2 = np.full(len(F), 2, np.int8)
        for op, i1, i2, j1, j2 in SequenceMatcher(None, Fi.tolist(), l2.tolist(), autojunk=False).get_opcodes():
            if op == 'equal': te2[i1:i2] = tl2[j1:j2]
        TEV[tau_] = te2
    TRN = []
    if u in KEYU and len(enc) > 1:
        kt = EKT[u]; tv = np.array(times, dtype='datetime64[ns]'); contents = list(g_.writing_content)
        for k in range(1, len(enc)):
            prv, cur = enc[k - 1], enc[k]; L = len(cur)
            m = min(len(prv), L); ne = np.flatnonzero(prv[:m] != cur[:m]); pre_ = int(ne[0]) if len(ne) else m
            mm = m - pre_; ne2 = np.flatnonzero(prv[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
            ins = L - suf - pre_
            if ins <= 0: continue
            itoks = tok(contents[k])[pre_:L - suf]; nch = sum(len(t_) for t_ in itoks) + max(len(itoks) - 1, 0)
            nk = int(np.searchsorted(kt, tv[k] + np.timedelta64(1, 's'), 'right') - np.searchsorted(kt, max(tv[k - 1], tv[k] - np.timedelta64(3, 's')), 'right'))
            TRN.append((ins, nch, nk))
    TK[u] = dict(F=F, lab=lab, var=var, tt=tt, times=times, te=te, TEV=TEV, TRN=TRN, cls=cls, FG=FG, FGs=FGs, ted=sc['ted'], tbot=tbot, tpr=tpr)""")
cut = src.index('# keystroke support per labelled run')
sys.argv = [os.path.join(HERE, 'r9b_flora_token_entry.py'), D]
G = {'__name__': 'r9b_defs', '__file__': os.path.join(HERE, 'r9b_flora_token_entry.py'), 'TAUS': TAUS}
exec(compile(src[:cut], 'r9b_flora_token_entry.py', 'exec'), G)
TK, users, KEYU, EKT, grams, tok, d = (G[k] for k in ('TK', 'users', 'KEYU', 'EKT', 'grams', 'tok', 'd'))
N = 8; NB = 2000; rng = np.random.default_rng(2301)
log('labels', len(users))
out = {}
share = lambda m: float(m.mean()) if len(m) else 0.0

# ---------- F-51 partial identification (X-with-C from src/r23_flora_xsub.py, the partition of Table 5)
SC = pd.read_csv(os.path.join(HERE, '..', 'results', 'flora_student_classes.csv')).set_index('user_id')
XW = pd.read_csv(os.path.join(HERE, '..', 'results', 'r23_flora_xwithc.csv')).set_index('user_id').X_with_C   # src/r23_flora_xsub.py
rows = []
for u in users:
    t = TK[u]; lab = t['lab']; F = t['F']; big = t['te'] == 2
    after = np.zeros(len(F), bool)
    for i, g in enumerate(t['FG']):
        if t['cls'][g] == 'E' and g in t['tbot']:
            te_ = t['ted'].get(t['FGs'][g])
            if te_ is not None and pd.Timestamp(te_) >= pd.Timestamp(t['tbot'][g]): after[i:i + N] = True
    E = lab == 'E'; Ea = E & after
    rows.append(dict(u=u, C=share(lab == 'C'), E=share(E), E_after=share(Ea), E_before=share(E & ~after), E_large=share(E & big),
                     E_before_large=share(E & ~after & big), S=share(lab == 'S'), T=share(t['var'] == 'T'), X=share(t['var'] == 'X'),
                     X_with_C=float(XW.get(u, 0.0)), C_released=float(SC.loc[u, 'C'])))
S = pd.DataFrame(rows).set_index('u')
S.drop(columns=['C_released']).to_csv(os.path.join(HERE, '..', 'results', 'r23_flora_students.csv'), float_format='%.6f')
out['F51_check_C_against_released_labels_max_abs_diff'] = float((S.C - S.C_released).abs().max())
def at(v, thr_):
    v = np.asarray(v); b = [np.mean(v[rng.integers(0, len(v), len(v))] >= thr_) for _ in range(NB)]
    return [float(np.mean(v >= thr_)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
ladder = [
    ('L0 lower: C only', 'A1-A3; no coincidence; E-after, X and T are not chatbot text', S.C),
    ('L0b lower, echo first in the token priority', 'as L0, with the projection E before C (Table 4)', None),
    ('L1 + E-after', 'A1-A3; echo that entered only after a reply containing it may have been taken from that reply', S.C + S.E_after),
    ('L2 + E-after + X-with-C', 'A1-A2; edited chatbot text that entered in the same paste as chatbot-first text', S.C + S.E_after + S.X_with_C),
    ('L3 + E-after + all X', 'A1-A2; any large unmatched insertion may be edited chatbot text', S.C + S.E_after + S.X),
    ('L4 + all E + all X', 'A2 only: replies not logged (A1 dropped), so echo that came first may also be chatbot text', S.C + S.E + S.X),
    ('L5 + all E + all X + T', 'no assumption: typed text may be chatbot text retyped with changes', S.C + S.E + S.X + S['T'] + S['S'])]
F51 = {'mean_shares': {k: float(S[k].mean()) for k in ['C', 'E', 'E_after', 'E_before', 'E_large', 'E_before_large', 'S', 'T', 'X', 'X_with_C']}, 'ladder': []}
for name, assume, v in ladder:
    if v is None: continue
    F51['ladder'].append({'step': name, 'assumptions': assume, 'mean': float(v.mean()), **{f'at_or_above_{int(q * 100)}': at(v, q) for q in (.25, .5, .75)}})
F51['table4_counts_for_reference'] = {'C_plus_large_E': at(S.C + S.E_large, .5), 'C_plus_X_with_C': at(S.C + S.X_with_C, .5),
                                      'C_plus_large_E_plus_X_with_C': at(S.C + S.E_large + S.X_with_C, .5)}
out['F51_partial_identification'] = F51
log('F51', [(r['step'], r['at_or_above_50']) for r in F51['ladder']])

# ---------- F-52 data-driven tau
T_ = pd.DataFrame([(u, *r) for u in KEYU for r in TK[u]['TRN']], columns=['u', 'ins', 'nch', 'nk'])
T_['typed'] = T_.nk >= .5 * T_.nch
ty = T_[T_.typed]
def wq(x, w, q):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]; c = np.cumsum(w) / w.sum(); return int(x[np.searchsorted(c, q)])
tau_tok = wq(ty.ins.values, ty.ins.values, .99); tau_ins = wq(ty.ins.values, np.ones(len(ty)), .99)
by = T_.groupby(T_.ins.clip(upper=21)).typed.agg(['size', 'mean'])
grid = {}
for tau_ in TAUS:
    Xs = np.array([share(np.isin(TK[u]['lab'], ['C', 'E', 'S'], invert=True) & (TK[u]['TEV'][tau_] == 2)) for u in users])
    Ts = np.array([share(np.isin(TK[u]['lab'], ['C', 'E', 'S'], invert=True) & (TK[u]['TEV'][tau_] == 1)) for u in users])
    grid[tau_] = {'mean_T': float(Ts.mean()), 'mean_X': float(Xs.mean()), 'C_plus_X_with_C_unchanged': True}
# student-bootstrap interval of tau_hat (token-weighted)
us = ty.u.unique(); gr = {k: v.ins.values for k, v in ty.groupby('u')}; bt = []
for _ in range(500):
    s_ = rng.choice(us, len(us)); x = np.concatenate([gr[k] for k in s_]); bt.append(wq(x, x, .99))
out['F52_tau_rule'] = {'keystroke_students': int(T_.u.nunique()), 'insertions': int(len(T_)), 'typed_insertions': int(len(ty)),
                       'tau_hat_token_weighted_p99': tau_tok, 'tau_hat_token_weighted_p99_bootstrap_2_5_97_5': [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                       'tau_hat_insertion_weighted_p99': tau_ins,
                       'typed_tokens_in_insertions_le_tau': {k: float(ty.ins[ty.ins <= k].sum() / ty.ins.sum()) for k in (1, 2, 3, 4, 5, 6, 8, 10, 13, 20)},
                       'typed_share_by_size': {int(k): {'insertions': int(r['size']), 'typed': float(r['mean'])} for k, r in by.iterrows()},
                       'grid': grid}
log('F52', tau_tok, tau_ins, grid[5], grid.get(tau_tok))

# ---------- trace for F-53 and F-54 (as src/r9_flora.py)
tr = pd.read_pickle(f'{D}/trace.pkl'); tr['t'] = pd.to_datetime(tr.log_trace_time, unit='ms')
ek = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy()
sel = tr[(tr.learning_action_label == 'READ_FEEDBACK_MULTI_AGENTS') & tr.event_value.astype(str).str.startswith('MOUSE_SELECT:::')][['user_id', 't', 'event_value']].copy()
win = tr[tr.learning_action_label == 'OPEN_CHATGPT_MULTI_AGENTS_SINGLE_WINDOW'][['user_id', 't']].copy()
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); del tr
pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
ek = ek.sort_values(['user_id', 't']); v = ek.event_value.astype(str).str.replace('KEY:::', '', regex=False); key = v.str.split('---').str[0]
ek['mod'] = key.isin(['Control', 'Meta']).values; ek['isv'] = v.str.endswith('KeyV').values
EPS = {}
for u, x in ek.groupby('user_id'):
    res = []; lastmod = None
    for t_, m_, v_ in zip(x.t.values, x['mod'].values, x.isv.values):
        if m_: lastmod = t_
        elif v_ and lastmod is not None and (t_ - lastmod) / np.timedelta64(1, 's') <= 1.5: res.append(t_)
    EPS[u] = np.array(res, dtype='datetime64[ns]')
SEL = {u: [(t_, set(grams(tok(s.replace('MOUSE_SELECT:::', '')), N))) for t_, s in zip(x.t, x.event_value.astype(str))] for u, x in sel.groupby('user_id')}
PGT = {u: [(np.datetime64(t_), set(grams(tok(s), N))) for t_, s in zip(x.t, x.txt)] for u, x in pa.groupby('user_id')}
WIN = {u: np.sort(x.t.values) for u, x in win.groupby('user_id')}

def evidence(u, i, j):
    t = TK[u]; FG = t['FG']
    gs = [FG[k] for k in range(i, max(i, j - N + 1)) if t['ted'].get(t['FGs'][FG[k]]) is not None]
    tes = [t['ted'][t['FGs'][g]] for g in gs]
    if not tes: return None
    t1, t2 = np.datetime64(min(tes)) - np.timedelta64(3, 's'), np.datetime64(max(tes)) + np.timedelta64(1, 's')
    chars = len(' '.join(t['F'][i:j])); ekt = EKT.get(u, np.array([], 'datetime64[ns]')); keys = int(((ekt >= t1) & (ekt <= t2)).sum())
    eps = EPS.get(u, np.array([], 'datetime64[ns]')); pasted = bool(((eps >= t1 - np.timedelta64(5, 's')) & (eps <= t2)).sum())
    pg = set(gs)
    selb = any(pd.Timestamp(ts) <= pd.Timestamp(t2) and (sg & pg) for ts, sg in SEL.get(u, []))
    inpaste = any(tp_ <= t2 and (sg & pg) for tp_, sg in PGT.get(u, []))
    tb = min(pd.Timestamp(t['tbot'][g]) for g in pg if g in t['tbot']) if any(g in t['tbot'] for g in pg) else None
    w_ = WIN.get(u, np.array([], 'datetime64[ns]'))
    window = bool(tb is not None and ((w_ <= t2) & (w_ >= np.datetime64(tb) - np.timedelta64(3600, 's'))).any())
    return dict(keys=keys, chars=chars, typed=keys >= .5 * chars and not pasted, not_typed=keys < .2 * chars, pasted_shortcut=pasted,
                selected=selb, in_logged_paste=inpaste, has_logged_pastes=u in PGT, chat_window_opened_near=window)

def runs_of(L, labels, minlen=8):
    r = []; i = 0
    while i < len(L):
        j = i
        while j < len(L) and L[j] == L[i]: j += 1
        if j - i >= minlen and L[i] in labels: r.append((L[i], i, j))
        i = j
    return r
def boot_share(vals, groups):
    vals = np.asarray(vals, float); groups = np.asarray(groups); us_ = np.unique(groups); idx = {q: np.flatnonzero(groups == q) for q in us_}; bs = []
    for _ in range(NB):
        b = rng.choice(us_, len(us_)); ii = np.concatenate([idx[q] for q in b]); bs.append(vals[ii].mean())
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

# ---------- F-53 census of chatbot-first runs
cr = []
for u in KEYU:
    for c, i, j in runs_of(TK[u]['lab'], 'C'):
        ev = evidence(u, i, j)
        if ev: cr.append(dict(u=u, i=i, j=j, tokens=j - i, **ev))
cr = pd.DataFrame(cr)
F53 = {'runs': int(len(cr)), 'students': int(cr.u.nunique()), 'tokens': int(cr.tokens.sum())}
for k in ('not_typed', 'typed', 'pasted_shortcut', 'selected', 'in_logged_paste'):
    F53[k] = boot_share(cr[k], cr.u)
F53['selected_or_pasted_shortcut'] = boot_share(cr.selected | cr.pasted_shortcut, cr.u)
F53['token_weighted'] = {k: float((cr[k] * cr.tokens).sum() / cr.tokens.sum()) for k in ('not_typed', 'typed', 'selected', 'pasted_shortcut')}
F53['students_with_logged_pastes'] = {'runs': int(cr.has_logged_pastes.sum()), 'in_logged_paste': boot_share(cr[cr.has_logged_pastes].in_logged_paste, cr[cr.has_logged_pastes].u)}
F53['not_selected'] = {'runs': int((~cr.selected).sum()), 'typed': float(cr[~cr.selected].typed.mean()), 'pasted_shortcut': float(cr[~cr.selected].pasted_shortcut.mean()),
                       'chat_window_opened_near': float(cr[~cr.selected].chat_window_opened_near.mean()),
                       'neither_typed_nor_shortcut': float((~cr[~cr.selected].typed & ~cr[~cr.selected].pasted_shortcut).mean())}
out['F53_C_census'] = F53
log('F53', F53)

# ---------- F-54 the seven non-selected C passages of the R9 audit
au = pd.read_csv(os.path.join(HERE, '..', 'results', 'r9_audit_sample.csv'))
au = au[(au.cls == 'C') & (~au.selected_in_chat_before_entry.astype(bool))]
F54 = []
for r in au.itertuples():
    ev = evidence(int(r.u), int(r.i), int(r.j)) or {}
    F54.append({'tokens': int(r.j - r.i), 'keys': int(r.keys), 'chars': int(r.chars), 'pasted_shortcut': bool(r.pasted_shortcut),
                'in_logged_paste': ev.get('in_logged_paste'), 'has_logged_pastes': ev.get('has_logged_pastes'),
                'chat_window_opened_near': ev.get('chat_window_opened_near'), 'selected_recomputed': ev.get('selected')})
out['F54_nonselected_C_audit_passages'] = F54
log('F54', F54)
out['seconds'] = round(time.perf_counter() - t00)
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r23_flora.json'), 'w'), indent=1, default=float)
log('done')
