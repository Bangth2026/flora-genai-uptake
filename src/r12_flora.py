"""Analysis log R12, F-6 to F-10 (FLoRA): unresolved tokens that are absent from the last snapshot (entry size
infinite), length of the working span and timing of unresolved text, share of students without uptake, naive overlap
at n = 5 and 12, and paste comparisons that account for pastes nested in students.
Classes are read from results/flora_token_classes.csv.gz (log entry R11, token entry rule); entry times are recomputed
with the token entry propagation of src/r11_flora.py.
Usage: python -I src/r12_flora.py DATA_DIR -> results/r12_flora.json
DATA_DIR needs dialogue.xlsx, proposal.xlsx, writing.pkl and pastes.pkl (rows of the system trace log, Figshare
files 5.2, with learning_action_label PASTE_TEXT_ESSAY; columns user_id, log_trace_time, event_value)."""
import sys, re, json, warnings
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2120); NB = 2000
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
users = sorted(set(d.user_id) & set(p.index))
P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])
CL = {u: g['class'].values for u, g in tc.groupby('user_id')}
assert all(len(CL[u]) == len(P[u]) for u in users)
sc = pd.read_csv('results/flora_student_classes.csv').set_index('user_id').loc[users]
o = {'n_students': len(users)}

# ---------- F-8 share of students without uptake; shape of the U distribution
U = sc.C.values
o['F8_U_distribution'] = {'mean': float(U.mean()), 'median': float(np.median(U)), 'share_U_zero': float(np.mean(U == 0)),
                          'share_U_below_1pct': float(np.mean(U < .01)), 'quartiles': np.percentile(U, [25, 50, 75]).tolist(),
                          'share_U_ge50': float(np.mean(U >= .5)), 'skewness': float(stats.skew(U))}

# ---------- F-9 naive overlap at n = 5, 8, 12
REP = {u: [tok(s) for s in g.chatbot_utterance_text.astype(str)] for u, g in d.groupby('user_id')}
o['F9_naive_overlap'] = {}
for n in (5, 8, 12):
    sh = []
    for u in users:
        G = {g for r in REP.get(u, []) for g in grams(r, n)}; F = P[u]; m = np.zeros(len(F), bool)
        for i, g in enumerate(grams(F, n)):
            if g in G: m[i:i + n] = True
        sh.append(m.mean() if len(F) else 0.0)
    sh = np.array(sh); bs = [np.mean(sh[rng.integers(0, len(sh), len(sh))] >= .5) for _ in range(NB)]
    o['F9_naive_overlap'][f'n{n}'] = {'median': float(np.median(sh)), 'mean': float(sh.mean()), 'share_ge50': float(np.mean(sh >= .5)),
                                      'ci_share_ge50': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
print('F8-F9', o['F8_U_distribution'], o['F9_naive_overlap'], flush=True)

# ---------- token entry propagation (as src/r11_flora.py)
def propagate(enc, tau=5):
    prev = np.zeros(0, np.int32); lab = np.zeros(0, np.int8); tin = np.zeros(0, np.int64)
    for k, cur in enumerate(enc):
        L = len(cur)
        if len(prev) == 0:
            lab = np.full(L, 1 if L <= tau else 2, np.int8); tin = np.full(L, k, np.int64); prev = cur; continue
        m = min(len(prev), L); ne = np.flatnonzero(prev[:m] != cur[:m]); pre = int(ne[0]) if len(ne) else m
        mm = min(len(prev), L) - pre; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
        a, b = prev[pre:len(prev) - suf], cur[pre:L - suf]
        if len(a) == 0 and len(b) == 0: prev = cur; continue
        nl = np.empty(L, np.int8); nt = np.empty(L, np.int64)
        nl[:pre] = lab[:pre]; nt[:pre] = tin[:pre]
        if suf: nl[L - suf:] = lab[len(prev) - suf:]; nt[L - suf:] = tin[len(prev) - suf:]
        if len(b):
            ops = [('insert', 0, 0, 0, len(b))] if len(a) == 0 else SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes()
            for op, i1, i2, j1, j2 in ops:
                if op == 'equal': nl[pre + j1:pre + j2] = lab[pre + i1:pre + i2]; nt[pre + j1:pre + j2] = tin[pre + i1:pre + i2]
                elif op in ('insert', 'replace'): nl[pre + j1:pre + j2] = 1 if (j2 - j1) <= tau else 2; nt[pre + j1:pre + j2] = k
        lab, tin, prev = nl, nt, cur
    return prev, lab, tin

w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
TK = {}; rows = []
for kk, (u, g_) in enumerate(w[w.user_id.isin(users)].groupby('user_id')):
    times = list(g_.t); vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in g_.writing_content]
    F = P[u]; Fi = np.array([vocab.setdefault(x, len(vocab)) for x in F], np.int32)
    last, tl, ti = propagate(enc)
    tt = np.full(len(F), -1, np.int64)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, Fi.tolist(), last.tolist(), autojunk=False).get_opcodes():
        if op == 'equal': tt[i1:i2] = ti[j1:j2]
    TK[u] = dict(tt=tt, times=times)
    if kk % 50 == 0: print(kk, flush=True)
del w
miss = [u for u in users if u not in TK]
for u in miss: TK[u] = dict(tt=np.full(len(P[u]), -1, np.int64), times=[])

# ---------- F-6 unresolved tokens absent from the last snapshot (entry size infinite)
xn = sum(int((CL[u] == 'X').sum()) for u in users); xinf = sum(int(((CL[u] == 'X') & (TK[u]['tt'] < 0)).sum()) for u in users)
per = np.array([((CL[u] == 'X') & (TK[u]['tt'] < 0)).mean() for u in users])
o['F6_X_not_in_last_snapshot'] = {'share_of_X_tokens_pooled': xinf / xn, 'mean_share_of_proposal': float(per.mean()),
                                  'median_share_of_proposal': float(np.median(per)), 'students_with_any': int((per > 0).sum()),
                                  'students_with_ge10pct_of_proposal': int((per >= .10).sum()), 'students_without_snapshots': len(miss),
                                  'all_tokens_not_in_last_snapshot_mean_share': float(np.mean([(TK[u]['tt'] < 0).mean() for u in users]))}
print('F6', o['F6_X_not_in_last_snapshot'], flush=True)

# ---------- F-7 working span and timing of C, T, X
spans = []; acc = {c: Counter() for c in 'CTX'}
for u in users:
    times = TK[u]['times']
    if not times: continue
    T0, T1 = times[0], times[-1]; span = (T1 - T0).total_seconds(); spans.append(span / 86400)
    tt = TK[u]['tt']; cl = CL[u]
    for c in 'CTX':
        m = (cl == c) & (tt >= 0)
        if not m.any(): continue
        et = np.array([(times[k] - T0).total_seconds() for k in tt[m]]); rel = et / span if span > 0 else np.zeros(len(et))
        to_end = np.array([(T1 - times[k]).total_seconds() for k in tt[m]])
        acc[c]['n'] += int(m.sum()); acc[c]['last24'] += int((to_end <= 86400).sum()); acc[c]['first_q'] += int((rel < .25).sum())
        acc[c]['last_q'] += int((rel >= .75).sum()); acc[c]['first_snapshot'] += int((tt[m] == 0).sum())
        acc[c]['first_hour'] += int((et <= 3600).sum()); acc[c]['first24'] += int((et <= 86400).sum())
        if span > 4 * 86400: acc[c]['n_long'] += int(m.sum()); acc[c]['last24_long'] += int((to_end <= 86400).sum())
spans = np.array(spans)
o['F7_working_span_days'] = {'definition': 'time from the first to the last editor snapshot of the student',
                             'n': int(len(spans)), 'quartiles': np.percentile(spans, [25, 50, 75]).tolist(), 'p90': float(np.percentile(spans, 90)),
                             'share_le_1day': float(np.mean(spans <= 1)), 'share_le_4days': float(np.mean(spans <= 4)), 'max': float(spans.max())}
o['F7_timing_by_class'] = {c: {'last24h': v['last24'] / v['n'], 'first_quarter_of_span': v['first_q'] / v['n'], 'last_quarter_of_span': v['last_q'] / v['n'],
                               'first_snapshot': v['first_snapshot'] / v['n'], 'first_hour': v['first_hour'] / v['n'], 'first_24h': v['first24'] / v['n'],
                               'last24h_students_span_gt4days': v['last24_long'] / v['n_long'] if v['n_long'] else None, 'tokens': v['n']} for c, v in acc.items()}
print('F7', o['F7_working_span_days'], o['F7_timing_by_class'], flush=True)

# ---------- F-10 logged pastes, C-dominated against X-dominated, with students as clusters
pa = pd.read_pickle(f'{D}/pastes.pkl'); pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms')
pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
REPT = {u: np.sort(pd.to_datetime(x.chatbot_utterance_time).values) for u, x in d.groupby('user_id')}
PR = []
for u in sorted(set(pa.user_id) & set(users)):
    x = pa[pa.user_id == u]; F = P[u]; var = CL[u]; Fg = {}
    for i, g in enumerate(grams(F, 8)): Fg.setdefault(g, []).append(i)
    T1 = TK[u]['times'][-1] if TK[u]['times'] else None; rt = REPT.get(u, np.array([], 'datetime64[ns]'))
    for t_, s_ in zip(x.t, x.txt):
        pt = tok(s_)
        if len(pt) < 8: continue
        cov = set()
        for g in grams(pt, 8):
            for i in Fg.get(g, []): cov.update(range(i, i + 8))
        if len(cov) < 8: continue
        cc = Counter(var[i] for i in cov); n_ = len(cov); shC, shX = cc['C'] / n_, cc['X'] / n_
        grp = 'C' if shC >= .8 else ('X' if shX >= .8 else ('mixed_CX' if shC >= .1 and shX >= .1 and shC + shX >= .8 else 'other'))
        prev_r = rt[rt <= np.datetime64(t_)]; since = (np.datetime64(t_) - prev_r[-1]) / np.timedelta64(1, 's') if len(prev_r) else np.nan
        PR.append(dict(u=u, grp=grp, len=len(pt), h_before_end=(T1 - t_).total_seconds() / 3600 if T1 is not None else np.nan, s_since_reply=since))
PR = pd.DataFrame(PR); cx = PR[PR.grp.isin(['C', 'X'])].copy(); cx['isX'] = (cx.grp == 'X').astype(int)
o['F10_pastes_counts'] = {g: {'n': int(len(x)), 'students': int(x.u.nunique())} for g, x in PR.groupby('grp')}
o['F10_students_with_both_C_and_X_pastes'] = int(len(set(cx[cx.grp == 'C'].u) & set(cx[cx.grp == 'X'].u)))
T = {}
for k, off in (('len', 0), ('h_before_end', 1 / 60), ('s_since_reply', 1)):
    z = cx.dropna(subset=[k]).copy(); z['y'] = np.log(z[k].clip(lower=0) + off)
    mw = float(stats.mannwhitneyu(z[z.isX == 1][k], z[z.isX == 0][k]).pvalue)
    try:
        mm = smf.mixedlm('y ~ isX', z, groups=z.u).fit(reml=True); coef, pv = float(mm.params['isX']), float(mm.pvalues['isX'])
        ci = [float(x) for x in mm.conf_int().loc['isX']]
    except Exception as e:
        coef = pv = None; ci = None
    # cluster bootstrap of the difference in medians (X minus C), resampling students
    studs = z.u.unique(); idx = {s: np.where(z.u.values == s)[0] for s in studs}; bs = []
    for _ in range(NB):
        b = np.concatenate([idx[s] for s in rng.choice(studs, len(studs))]); zz = z.iloc[b]
        a1, a0 = zz[zz.isX == 1][k], zz[zz.isX == 0][k]
        if len(a1) and len(a0): bs.append(a1.median() - a0.median())
    diff = float(z[z.isX == 1][k].median() - z[z.isX == 0][k].median())
    T[k] = {'median_X': float(z[z.isX == 1][k].median()), 'median_C': float(z[z.isX == 0][k].median()), 'mannwhitney_p_pastes_as_independent': mw,
            'mixedlm_log_coef_X': coef, 'mixedlm_p': pv, 'mixedlm_ci': ci, 'ratio_X_over_C_from_mixedlm': float(np.exp(coef)) if coef is not None else None,
            'median_diff_X_minus_C': diff, 'cluster_bootstrap_ci_median_diff': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
T['within_60s_of_reply'] = {'X': float((cx[cx.isX == 1].s_since_reply <= 60).mean()), 'C': float((cx[cx.isX == 0].s_since_reply <= 60).mean())}
o['F10_pastes_C_vs_X'] = T
print('F10', json.dumps(T, indent=0, default=float), flush=True)
json.dump(o, open('results/r12_flora.json', 'w'), indent=1, default=float)
print('done')
