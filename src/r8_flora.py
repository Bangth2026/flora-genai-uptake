"""Plan R8, FLoRA (ANALYSIS_LOG_R8.md J1-J5): misclassification sensitivity of the grade model (C + lambda*X, C with the
untraced text that entered together with chatbot text), equivalence tests, order of the E and S rules, students
contributing to the detector comparison, and a token-level export of the five provenance classes.
Readers, scan and classification are copied from src/r6_flora.py (plan R4 rules, n = 8, tau = 5, delta = 0).
Usage: python -I src/r8_flora.py DATA_DIR [R2_PASSAGES_PKL] -> results/r8_flora.json, results/flora_token_classes.csv.gz,
       results/flora_student_classes.csv"""
import sys, re, json, warnings, time
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from scipy import stats, optimize
import statsmodels.api as sm
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2080); NB = 2000
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
an = pd.read_pickle(f'{D}/annot.pkl'); tr = pd.read_pickle(f'{D}/trace.pkl')
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); del tr
pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms'); pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
users = sorted(set(d.user_id) & set(p.index))
LIMIT = None
P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
users_all = users; users = users[:LIMIT] if LIMIT else users
score = p.loc[users, 'proposal_score'].values.astype(float)
NS = (5, 8, 12)
H = {n: set(g for t in an.highlight_text.dropna() for g in grams(tok(t), n)) for n in NS}
Cc = {n: Counter(g for u in users_all for g in set(grams(P[u], n))) for n in NS}
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
WG = {u: g for u, g in w.groupby('user_id')}
U1 = np.load('results/U_R1.npy')[:len(users)]
NS = (8,)

def dialogue(u, n):
    x = d[d.user_id == u]; tbot, tpr, plen = {}, {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s), n): tbot.setdefault(g, pd.Timestamp(t))
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(s), n): tpr.setdefault(g, pd.Timestamp(t))
    return tbot, tpr

def scan(enc, times, idx, n, need, want_seen=False):
    """Entry of each n-gram of A (int-encoded tuples in need): time, insertion count of the whole transition,
    size of the inserted segment containing it, transition index. Optionally first-seen time of every n-gram."""
    need = set(need); ted, ins, seg, tid, seen = {}, {}, {}, {}, {}
    prev = np.zeros(0, np.int32)
    for k, i in enumerate(idx):
        if not need and not want_seen: break
        cur = enc[i]; L = len(cur)
        if len(prev) == 0:
            segs = [(0, L)] if L else []; lo, hi = 0, L - n + 1
        else:
            m = min(len(prev), L); ne = np.flatnonzero(prev[:m] != cur[:m]); pre = int(ne[0]) if len(ne) else m
            mm = min(len(prev), L) - pre; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
            a, b = prev[pre:len(prev) - suf], cur[pre:L - suf]
            if len(a) == 0 and len(b) == 0: prev = cur; continue
            if len(b) == 0: segs = []
            elif len(a) == 0 or len(b) == 1: segs = [(pre, pre + len(b))]
            else:
                segs = [(pre + j1, pre + j2) for op, i1, i2, j1, j2 in SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes() if op in ('insert', 'replace')]
            lo, hi = max(0, pre - n + 1), min(L - n + 1, max(L - suf, pre + 1))
        tot = sum(j2 - j1 for j1, j2 in segs); t = times[i]
        for s in range(max(lo, 0), max(hi, 0)):
            g = tuple(cur[s:s + n].tolist())
            if want_seen and g not in seen: seen[g] = t
            if g in need:
                best = (0, 0)
                for j1, j2 in segs:
                    ov = min(s + n, j2) - max(s, j1)
                    if ov > best[0]: best = (ov, j2 - j1)
                ted[g] = t; ins[g] = tot; seg[g] = best[1]; tid[g] = k; need.discard(g)
        prev = cur
    return dict(ted=ted, ins=ins, seg=seg, tid=tid, seen=seen)

def classify(FGs, tbot, tpr, sc, n, tau=5, delta=pd.Timedelta(0), mode='ins', k_peer=5):
    cls = {}
    for g, gi in FGs.items():
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]; shared = g in H[n] or Cc[n][g] - 1 >= k_peer
        te = sc['ted'].get(gi)
        if bot and not echo and not shared and te is not None and tbot[g] <= te + delta: cls[g] = 'C'
        elif bot and (echo or (te is not None and te < tbot[g])): cls[g] = 'E'
        elif shared: cls[g] = 'S'
        elif te is None: cls[g] = 'X'
        else: cls[g] = 'T' if sc[mode][gi] <= tau else 'X'
    return cls

def project(F, n, cls, order='CESTX'):
    rank = {c: i for i, c in enumerate(order)}; best = np.full(len(F), 99)
    for i, g in enumerate(grams(F, n)): best[i:i + n] = np.minimum(best[i:i + n], rank[cls[g]])
    return np.array([order[b] if b < 99 else 'X' for b in best])

def majority(F, n, cls, order='CESTX'):
    votes = [Counter() for _ in F]; rank = {c: i for i, c in enumerate(order)}
    for i, g in enumerate(grams(F, n)):
        for j in range(i, i + n): votes[j][cls[g]] += 1
    return np.array([min(v, key=lambda c: (-v[c], rank[c])) if v else 'X' for v in votes])


import statsmodels.formula.api as smf
PASS = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None
LIMIT = None; users = users_all
LAB = {}; XC = {}; ES = Counter(); ESlab = {}
t0 = time.time()
for k, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in (g_.writing_content if g_ is not None else [])]
    F = P[u]; Fi = [vocab.setdefault(x, len(vocab)) for x in F]
    tbot, tpr = dialogue(u, 8); FG = grams(F, 8); FGs = {g: tuple(Fi[i:i + 8]) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), 8, set(FGs.values()))
    cls = classify(FGs, tbot, tpr, sc, 8); lab = project(F, 8, cls); LAB[u] = lab
    Ctid = {sc['tid'][FGs[g]] for g, c in cls.items() if c == 'C' and FGs[g] in sc['tid']}
    xc = np.zeros(len(F), bool)
    for i, g in enumerate(FG):
        if cls[g] == 'X' and FGs[g] in sc['tid'] and sc['tid'][FGs[g]] in Ctid: xc[i:i + 8] = True
    XC[u] = xc & (lab == 'X')
    # J3: S n-grams that were in the editor before the first reply containing them
    cls2 = dict(cls)
    for g, c in cls.items():
        if c != 'S': continue
        ES['S_grams'] += 1; te = sc['ted'].get(FGs[g])
        if g in tbot and te is not None and te < tbot[g]: cls2[g] = 'E'; ES['S_editor_before_reply'] += 1
    ESlab[u] = project(F, 8, cls2)
    if k % 25 == 0: print(k, round(time.time() - t0), flush=True)
del WG, w
out = {'n_students': len(users)}
sh = pd.DataFrame({c: [(LAB[u] == c).mean() for u in users] for c in 'CESTX'}, index=users)
sh['Xc'] = [XC[u].mean() for u in users]; sh['Xo'] = sh.X - sh.Xc
out['check_C_vs_U_R1'] = float(np.abs(sh.C.values - np.load('results/U_R1.npy')).max())
out['shares'] = {c: float(sh[c].mean()) for c in sh.columns}
out['Xc_share_of_X_tokens'] = float(sum(XC[u].sum() for u in users) / sum((LAB[u] == 'X').sum() for u in users))
out['students_with_any_Xc'] = float((sh.Xc > 0).mean())
out['rho_C_Xc'] = float(stats.spearmanr(sh.C, sh.Xc)[0]); out['rho_C_X'] = float(stats.spearmanr(sh.C, sh.X)[0])
# covariates as in src/r5_flora_grade.py
first = lambda df: df.drop_duplicates('user_id', keep='first').set_index('user_id')
lvl = lambda x: float(re.match(r"\s*(\d)", str(x)).group(1)) if re.match(r"\s*(\d)", str(x)) else np.nan
bio, pri = (first(pd.read_excel(f'{D}/{f}.xlsx')) for f in ['bio', 'prior'])
df = sh.copy(); df['score'] = p.loc[users, 'proposal_score'].values.astype(float)
df['prior_ds'] = pd.to_numeric(pri.reindex(users).has_prior_knowledge_in_DS, errors='coerce')
df['genai_fam'] = pri.reindex(users).extent_of_familiar_genai.map(lvl)
mt = bio.reindex(users).mother_tongue
df['english_l1'] = mt.astype(str).str.strip().str.lower().eq('english').astype(float).where(mt.notna())
df['master'] = bio.reindex(users).degree.astype(str).str.contains('Master').astype(float).where(bio.reindex(users).degree.notna())
COV = ['prior_ds', 'genai_fam', 'english_l1', 'master']; df[COV] = df[COV].fillna(df[COV].median())
z = lambda v: (v - v.mean()) / v.std()
gsd = float(df.score.std(ddof=0)); EQ = 0.25 * gsd
def fit(cols, adj):
    d_ = df.copy()
    for c in cols: d_[c + '_z'] = z(d_[c] if isinstance(c, str) else None)
    f = smf.ols('score ~ ' + ' + '.join(c + '_z' for c in cols) + (' + ' + ' + '.join(COV) if adj else ''), d_).fit(cov_type='HC3')
    return f
def tost(b, se):
    p = max(stats.norm.sf((b + EQ) / se), stats.norm.cdf((b - EQ) / se))
    return {'bound_points': EQ, 'p_tost': float(p), 'ci90': [float(b - 1.645 * se), float(b + 1.645 * se)],
            'smallest_bound_points': float(abs(b) + 1.645 * se), 'smallest_bound_grade_sd': float((abs(b) + 1.645 * se) / gsd)}
J1 = {}
for lam in (0, 0.25, 0.5, 1):
    df[f'comp{lam}'.replace('.', 'p')] = df.C + lam * df.X
for lam in (0.5, 1):
    df[f'compc{lam}'.replace('.', 'p')] = df.C + lam * df.Xc
for name in ['comp0', 'comp0p25', 'comp0p5', 'comp1', 'compc0p5', 'compc1']:
    for adj in (False, True):
        f = fit([name], adj); v = name + '_z'; b, se = float(f.params[v]), float(f.bse[v])
        J1[f'{name}|{"b" if adj else "a"}'] = {'per_sd': b, 'ci': [float(c) for c in f.conf_int().loc[v]], 'p': float(f.pvalues[v]), 'se': se, 'tost': tost(b, se),
                                                 'rho_with_C': float(stats.spearmanr(df[name], df.C)[0])}
for adj in (False, True):
    f = fit(['C', 'Xc'], adj)
    J1[f'two_component|{"b" if adj else "a"}'] = {v: {'per_sd': float(f.params[v + '_z']), 'ci': [float(c) for c in f.conf_int().loc[v + '_z']], 'p': float(f.pvalues[v + '_z'])} for v in ('C', 'Xc')}
    f = fit(['C', 'Xc', 'Xo'], adj)
    J1[f'three_component|{"b" if adj else "a"}'] = {v: {'per_sd': float(f.params[v + '_z']), 'ci': [float(c) for c in f.conf_int().loc[v + '_z']], 'p': float(f.pvalues[v + '_z'])} for v in ('C', 'Xc', 'Xo')}
out['J1_J2'] = J1; out['grade_sd'] = gsd
# J3
tot = sum(len(P[u]) for u in users)
out['J3_E_before_S'] = {**dict(ES), 'tokens_changed_share': sum(int((ESlab[u] != LAB[u]).sum()) for u in users) / tot,
                        'class_means_E_first': {c: float(np.mean([(ESlab[u] == c).mean() for u in users])) for c in 'CESTX'}}
# J4 students contributing to the detector comparison
if PASS:
    R = pd.read_pickle(PASS); R = R[R.user_id.isin(users)].copy()
    def pcl(u, text):
        t = tok(text); F = P[u]; n_ = len(t)
        for i in range(len(F) - n_ + 1):
            if F[i:i + n_] == t: return LAB[u][i:i + n_]
    pc = [pcl(u, s) for u, s in zip(R.user_id, R.text)]
    for c in 'TX': R[c] = [float((a == c).mean()) if a is not None else np.nan for a in pc]
    R['cls3'] = np.where(R.label == 'chatbot', 'chatbot', np.where((R.label == 'composed') & (R['T'] >= .9), 'typed', np.where((R.label == 'composed') & (R['X'] >= .9), 'untraced', 'other')))
    J4 = {'counts': R.cls3.value_counts().to_dict()}
    for c in ('chatbot', 'typed', 'untraced'):
        s_ = R[R.cls3 == c].user_id.value_counts()
        J4[c] = {'paragraphs': int(s_.sum()), 'students': int(len(s_)), 'max_by_one_student': int(s_.max()), 'max_share_one_student': float(s_.max() / s_.sum()),
                 'top3_share': float(s_.head(3).sum() / s_.sum()), 'median_per_student': float(s_.median())}
    out['J4_detector_students'] = J4

# J6 Tobit for the grade (censored at 15) with numerical-Hessian standard errors
from statsmodels.tools.numdiff import approx_hess
y = df.score.values; cen = y >= 15; X = np.c_[np.ones(len(df)), z(df.C).values]
def nllg(th):
    b, s = th[:2], np.exp(th[2]); mu = X @ b
    return -(stats.norm.logpdf(y[~cen], mu[~cen], s).sum() + stats.norm.logsf(15, mu[cen], s).sum())
f = optimize.minimize(nllg, np.r_[np.linalg.lstsq(X, y, rcond=None)[0], 0.0], method='BFGS')
se_n = np.sqrt(np.diag(np.linalg.inv(approx_hess(f.x, nllg)))); se_b = np.sqrt(np.diag(f.hess_inv))
out['J6_tobit15'] = {'b': float(f.x[1]), 'ci_numerical': [float(f.x[1] - 1.96 * se_n[1]), float(f.x[1] + 1.96 * se_n[1])], 'p': float(2 * stats.norm.sf(abs(f.x[1] / se_n[1]))),
                     'se_numerical': float(se_n[1]), 'se_bfgs': float(se_b[1])}
# J5 export
rows = []
for u in users:
    for i, (t_, c) in enumerate(zip(P[u], LAB[u])): rows.append((u, i, t_, c))
pd.DataFrame(rows, columns=['user_id', 'token_index', 'token', 'class']).to_csv('results/flora_token_classes.csv.gz', index=False, compression='gzip')
sh[['C', 'E', 'S', 'T', 'X', 'Xc']].rename(columns={'Xc': 'X_with_C'}).to_csv('results/flora_student_classes.csv', index_label='user_id', float_format='%.6f')
out['J5_export'] = {'tokens': len(rows), 'students': len(users)}
json.dump(out, open('results/r8_flora.json', 'w'), indent=1); print(json.dumps(out, indent=1))
