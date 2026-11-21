"""Analysis log R19, F-40 to F-43 (FLoRA).
F-40 the share of students at or above 25%, 50% and 75% under U and under five alternative counts (E before C in the
token priority; U plus large-entry echo; U plus all echo; U plus X-with-C; U plus large-entry echo plus X-with-C), with
student-bootstrap intervals, and the empirical distributions for the figure;
F-41 the threshold tau of the entry rule at 3, 5, 10 and 20: share at or above half of U plus large-entry echo, and the
mean shares of T and X;
F-42 the size of the insertion that brought each final token into the editor: share of tokens by size band, for all
tokens and for the tokens labelled T or X, and for X tokens the share whose insertion had keystroke support (printable
editor keystrokes from max(previous snapshot, snapshot - 3 s) to snapshot + 1 s at least half of the inserted characters);
F-43 completeness of the editor log for students with and without logged pastes: differences in medians and
standardised mean differences with student-bootstrap intervals (in place of the p-values of R10 P-2).
The attribution (src/r9b_flora_token_entry.py definitions, n = 8, tau = 5, delta = 0) does not change.
Usage: python -I src/r19_flora.py DATA_DIR R19_SUB_PKL -> results/r19_flora.json"""
import sys, os, json, time, pickle
import numpy as np, pandas as pd
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
D, SUBP = sys.argv[1], sys.argv[2]
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)
src = open(os.path.join(HERE, 'r9b_flora_token_entry.py')).read()
cut = src.index('# ======================= R9b body')
sys.argv = [os.path.join(HERE, 'r9b_flora_token_entry.py'), D]
G = {'__name__': 'r9b_defs', '__file__': os.path.join(HERE, 'r9b_flora_token_entry.py')}
exec(compile(src[:cut], 'r9b_flora_token_entry.py', 'exec'), G)
d, users, P, WG, tok, grams, scan, classify, project, dialogue = (G[k] for k in
    ('d', 'users_all', 'P', 'WG', 'tok', 'grams', 'scan', 'classify', 'project', 'dialogue'))
SUB = pickle.load(open(SUBP, 'rb'))
NB = 2000; rng = np.random.default_rng(2191); TAUS = (3, 5, 10, 20); INF = 10 ** 9
tr = pd.read_pickle(f'{D}/trace.pkl'); tr['t'] = pd.to_datetime(tr.log_trace_time, unit='ms')
ek = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy()
pa_users = set(tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].user_id); del tr
key = ek.event_value.astype(str).str.replace('KEY:::', '', regex=False).str.split('---').str[0]
EKT = {u: np.sort(g_.t.values[key[g_.index].str.len().eq(1).values]) for u, g_ in ek.groupby('user_id')}

def propagate_size(enc, times, kt, vocab_inv):
    """Per token of the last snapshot: size of the inserted or replaced segment that brought it in, and whether that
    transition had keystroke support (None when no keystrokes are logged for the student)."""
    prev = np.zeros(0, np.int32); size = np.zeros(0, np.int64); typ = np.zeros(0, np.int8)
    tv = np.array(times, dtype='datetime64[ns]')
    for k, cur in enumerate(enc):
        L = len(cur)
        if len(prev) == 0:
            if L:
                nch = sum(len(vocab_inv[x]) for x in cur) + L - 1
                ty = keyflag(kt, None, tv[k], nch)
                size = np.full(L, L, np.int64); typ = np.full(L, ty, np.int8)
            prev = cur; continue
        m = min(len(prev), L); ne = np.flatnonzero(prev[:m] != cur[:m]); pre = int(ne[0]) if len(ne) else m
        mm = min(len(prev), L) - pre; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
        a, b = prev[pre:len(prev) - suf], cur[pre:L - suf]
        if len(a) == 0 and len(b) == 0: prev = cur; continue
        ns = np.empty(L, np.int64); nt = np.empty(L, np.int8)
        ns[:pre] = size[:pre]; nt[:pre] = typ[:pre]
        if suf: ns[L - suf:] = size[len(prev) - suf:]; nt[L - suf:] = typ[len(prev) - suf:]
        if len(b):
            ops = [('insert', 0, 0, 0, len(b))] if len(a) == 0 else SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes()
            ins_ops = [(j1, j2) for op, i1, i2, j1, j2 in ops if op in ('insert', 'replace')]
            nch = sum(sum(len(vocab_inv[x]) for x in b[j1:j2]) + (j2 - j1 - 1) for j1, j2 in ins_ops)
            ty = keyflag(kt, tv[k - 1], tv[k], nch) if ins_ops else 0
            for op, i1, i2, j1, j2 in ops:
                if op == 'equal': ns[pre + j1:pre + j2] = size[pre + i1:pre + i2]; nt[pre + j1:pre + j2] = typ[pre + i1:pre + i2]
                elif op in ('insert', 'replace'): ns[pre + j1:pre + j2] = j2 - j1; nt[pre + j1:pre + j2] = ty
        size, typ, prev = ns, nt, cur
    return prev, size, typ

def keyflag(kt, t_prev, t_cur, nch):
    """1 typed, 0 not typed, -1 no keystroke log for the student"""
    if kt is None: return -1
    lo = t_cur - np.timedelta64(3, 's')
    if t_prev is not None and t_prev > lo: lo = t_prev
    n = int(np.searchsorted(kt, t_cur + np.timedelta64(1, 's'), 'right') - np.searchsorted(kt, lo, 'right'))
    return 1 if n >= 0.5 * nch else 0

rows = []; SZ = []; misalign = 0
for kk, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in (g_.writing_content if g_ is not None else [])]
    F = P[u]; Fi = np.array([vocab.setdefault(x, len(vocab)) for x in F], np.int32); inv = {v: k for k, v in vocab.items()}
    tbot, tpr = dialogue(u, 8); FG = grams(F, 8); FGs = {g: tuple(Fi[i:i + 8].tolist()) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), 8, set(FGs.values()))
    cls = classify(FGs, tbot, tpr, sc, 8); lab = project(F, 8, cls); labEC = project(F, 8, cls, order='ECSTX')
    kt = EKT.get(u)
    if enc: last, sz, ty = propagate_size(enc, times, kt, inv)
    else: last, sz, ty = np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(0, np.int8)
    size = np.full(len(F), INF, np.int64); typd = np.full(len(F), -1, np.int8)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, Fi.tolist(), last.tolist(), autojunk=False).get_opcodes():
        if op == 'equal': size[i1:i2] = sz[j1:j2]; typd[i1:i2] = ty[j1:j2]
    s_ = SUB.get(u)
    if s_ is not None and list(s_['A']) == list(F): xwc = np.array([x == 'X-with-C' for x in s_['sub']])
    else:
        misalign += 1; xwc = np.zeros(len(F), bool)
        if s_ is not None:
            for op, i1, i2, j1, j2 in SequenceMatcher(None, list(F), list(s_['A']), autojunk=False).get_opcodes():
                if op == 'equal': xwc[i1:i2] = [x == 'X-with-C' for x in s_['sub'][j1:j2]]
    xwc &= (lab == 'X') | (lab == 'T')
    sh = lambda m: float(m.mean()) if len(m) else 0.0
    r = dict(u=u, C=sh(lab == 'C'), C_EC=sh(labEC == 'C'), E=sh(lab == 'E'), Xwc=sh(xwc), paste=u in pa_users)
    for tau in TAUS:
        var = np.where(np.isin(lab, ['C', 'E', 'S']), lab, np.where(size <= tau, 'T', 'X'))
        r[f'Elarge_{tau}'] = sh((lab == 'E') & (size > tau)); r[f'T_{tau}'] = sh(var == 'T'); r[f'X_{tau}'] = sh(var == 'X')
    rows.append(r)
    var5 = np.where(np.isin(lab, ['C', 'E', 'S']), lab, np.where(size <= 5, 'T', 'X'))
    SZ.append(pd.DataFrame(dict(u=u, size=size, var=var5, typed=typd)))
    if kk % 50 == 0: log(kk)
S = pd.DataFrame(rows).set_index('u'); Z = pd.concat(SZ, ignore_index=True)
out = {'students': int(len(S)), 'subtype_alignment_fallbacks': misalign}

def thr_share(v, t):
    v = np.asarray(v); b = [np.mean(v[rng.integers(0, len(v), len(v))] >= t) for _ in range(NB)]
    return [float(np.mean(v >= t)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
counts = {'U': S.C, 'U_E_before_C': S.C_EC, 'U_plus_large_echo': S.C + S.Elarge_5, 'U_plus_all_echo': S.C + S.E,
          'U_plus_X_with_C': S.C + S.Xwc, 'U_plus_large_echo_plus_X_with_C': S.C + S.Elarge_5 + S.Xwc}
out['F40_thresholds'] = {k: {'mean': float(v.mean()), 'median': float(v.median()), **{f'at_or_above_{int(t * 100)}': thr_share(v, t) for t in (.25, .5, .75)}} for k, v in counts.items()}
out['F40_distributions'] = {k: sorted(float(x) for x in v) for k, v in counts.items()}
log('F40', {k: round(v['at_or_above_50'][0], 3) for k, v in out['F40_thresholds'].items()})
out['F41_tau'] = {str(t): {'U_plus_large_echo_at_or_above_half': thr_share(S.C + S[f'Elarge_{t}'], .5), 'mean_T': float(S[f'T_{t}'].mean()),
                           'mean_X': float(S[f'X_{t}'].mean()), 'mean_large_echo': float(S[f'Elarge_{t}'].mean())} for t in TAUS}
log('F41', {k: round(v['U_plus_large_echo_at_or_above_half'][0], 3) for k, v in out['F41_tau'].items()})
bands = [('1', 1, 1), ('2-5', 2, 5), ('6-20', 6, 20), ('>20', 21, INF - 1), ('not in last snapshot', INF, INF)]
def band_shares(z):
    return {b: float(((z['size'] >= lo) & (z['size'] <= hi)).mean()) for b, lo, hi in bands}
TX = Z[Z['var'].isin(['T', 'X'])]; XK = Z[(Z['var'] == 'X') & (Z.typed >= 0)]
out['F42_entry_size'] = {'all_tokens': band_shares(Z), 'T_or_X_tokens': band_shares(TX), 'X_tokens': band_shares(Z[Z['var'] == 'X']),
                         'X_tokens_keystroke_students': int(len(XK)), 'X_with_keystroke_support': float((XK.typed == 1).mean()),
                         'X_with_keystroke_support_by_band': {b: [float((XK.typed[(XK['size'] >= lo) & (XK['size'] <= hi)] == 1).mean()) if ((XK['size'] >= lo) & (XK['size'] <= hi)).any() else None,
                                                                  int(((XK['size'] >= lo) & (XK['size'] <= hi)).sum())] for b, lo, hi in bands}}
log('F42', out['F42_entry_size'])

# ---------- F-43 completeness of the editor log (measures of R10 P-2)
w = pd.read_pickle(f'{D}/writing.pkl'); w = w[w.user_id.isin(users)].sort_values(['user_id', 'writing_log_time'])
q = []
for u, g in w.groupby('user_id'):
    t = g.writing_log_time.values.astype(float) / 1e3; dt = np.diff(t); dts = dt[dt < 1800]
    prev = []; large = 0; changes = 0
    for c in g.writing_content.astype(str):
        cur = tok(c); m_ = min(len(prev), len(cur)); pre = 0
        while pre < m_ and prev[pre] == cur[pre]: pre += 1
        suf = 0
        while suf < m_ - pre and prev[-1 - suf] == cur[-1 - suf]: suf += 1
        ins = len(cur) - pre - suf
        if ins > 0: changes += 1; large += ins > 5
        prev = cur
    q.append(dict(u=u, paste=u in pa_users, snaps=len(g), per100=len(g) / max(len(P[u]), 1) * 100, med_dt=float(np.median(dts)) if len(dts) else np.nan,
                  sessions=int((dt >= 1800).sum()) + 1, large_share=large / max(changes, 1), writing_hours=float(dts.sum() / 3600)))
q = pd.DataFrame(q); F43 = {'n_with': int(q.paste.sum()), 'n_without': int((~q.paste).sum())}
for col in ('snaps', 'per100', 'med_dt', 'sessions', 'large_share', 'writing_hours'):
    a, b = q[q.paste][col].dropna().values, q[~q.paste][col].dropna().values
    def smd(x, y):
        sd = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)); return (x.mean() - y.mean()) / sd
    bm, bs = [], []
    for _ in range(NB):
        x, y = a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]; bm.append(np.median(x) - np.median(y)); bs.append(smd(x, y))
    F43[col] = {'median_with': float(np.median(a)), 'median_without': float(np.median(b)),
                'median_difference': [float(np.median(a) - np.median(b)), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                'smd': [float(smd(a, b)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
out['F43_paste_group_log_completeness'] = F43; log('F43', {k: [round(x, 2) for x in v['smd']] for k, v in F43.items() if isinstance(v, dict)})
out['seconds'] = round(time.perf_counter() - t00)
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r19_flora.json'), 'w'), indent=1)
log('done')
