"""Plan R4, part R: RelianceScope chance baseline, reference-solution and copy-based bounds, length-controlled
human-code validation, edit-size entry rule, censored post-test model.
Usage: python -I src/r4_rs.py RS_DIR PROMPT_TXT -> results/r4_rs_stats.json"""
import sys, json, re, warnings
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
D, PROMPT = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(2040); NB = 2000; J = 20; n = 8
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k=8: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
def cover(F, G, k=8):
    m = np.zeros(len(F), bool)
    for i, g in enumerate(grams(F, k)):
        if g in G: m[i:i + k] = True
    return m
L = lambda f: pd.DataFrame(list(json.load(open(f'{D}/logs/{f}')).values())[0])
msg, ed, conv = L('messages.json'), L('edit logs.json'), L('conversations.json')
cp = L('time and copy & paste logs.json'); cp = cp[cp.action == 'copied strings']
c2u = dict(zip(conv.id, conv.userId)); msg['u'] = msg.conversationId.map(c2u); ed['u'] = ed.conversationId.map(c2u)
ms = pd.read_csv(f'{D}/csv/measurement (by student).csv').set_index('Student ID')
cl = pd.read_csv(f'{D}/csv/chat logs (with classification).csv')
users = sorted(set(ms.index) & set(msg.u.dropna()) & set(ed.u.dropna()))
um = msg[(msg.role == 'user') & msg.snapshot.notna()].copy()
um['code'] = um.snapshot.map(lambda s: s.get('code') if isinstance(s, dict) else None)
S = pd.concat([ed[['u', 'createdAt', 'code']], um[['u', 'createdAt', 'code']]]).dropna().sort_values(['u', 'createdAt'])
template = Counter(ed.sort_values('createdAt').groupby('u').code.first()).most_common(1)[0][0]
TG = set(grams(tok(template)))
ptxt = open(PROMPT).read(); REF = set(grams(tok(ptxt[ptxt.index('<Answer Code>'):ptxt.index('</Answer Code>')])))
RANK = {'C': 0, 'E': 1, 'S': 2, 'T': 3, 'X': 4}; INV = {v: k for k, v in RANK.items()}

def student(u, rule='edit'):
    x = msg[msg.u == u].sort_values('createdAt'); s = S[S.u == u]; tbot, tpr = {}, {}
    for t, r, c in zip(x.createdAt, x.role, x.content):
        for g in grams(tok(c)): (tbot if r == 'assistant' else tpr).setdefault(g, t)
    F = tok(ed[ed.u == u].sort_values('createdAt').code.iloc[-1]); FG = grams(F); need = set(FG)
    ted, mode, prev = {}, {}, []
    for t, c in zip(s.createdAt, s.code):
        tk = tok(c); new = [g for g in set(grams(tk)) if g in need]
        if new:
            if rule == 'net': size = abs(len(tk) - len(prev))
            else: size = sum(j2 - j1 for op, i1, i2, j1, j2 in SequenceMatcher(None, prev, tk, autojunk=False).get_opcodes() if op in ('insert', 'replace'))
            for g in new: ted[g] = t; mode[g] = 'T' if size <= J else 'X'; need.discard(g)
        prev = tk
    cls = {}
    for g in set(FG):
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]
        if bot and not echo and g in ted and tbot[g] <= ted[g] and g not in TG: cls[g] = 'C'
        elif bot and (echo or (g in ted and ted[g] < tbot[g])): cls[g] = 'E'
        elif g in TG: cls[g] = 'S'
        else: cls[g] = mode.get(g, 'X')
    best = np.full(len(F), 9)
    for i, g in enumerate(FG): best[i:i + n] = np.minimum(best[i:i + n], RANK[cls[g]])
    return dict(F=F, lab=np.array([INV.get(b, 'X') for b in best]), cls=cls, tbot=tbot, ted=ted, mode=mode)

R = {u: student(u) for u in users}; Rn = {u: student(u, 'net') for u in users}
U = np.array([(R[u]['lab'] == 'C').mean() for u in users])
assert np.allclose(U, np.array([(Rn[u]['lab'] == 'C').mean() for u in users]))
bci = lambda f, v: [float(np.percentile([f(rng.choice(v, len(v))) for _ in range(NB)], q)) for q in (2.5, 97.5)]
out = {'n_students': len(users)}
for nm, RR in [('edit', R), ('net', Rn)]:
    sh = pd.DataFrame({c: [(RR[u]['lab'] == c).mean() for u in users] for c in 'CESTX'})
    out[f'classes_{nm}'] = {c: {'mean': float(sh[c].mean()), 'median': float(sh[c].median()), 'mean_ci': bci(np.mean, sh[c].values)} for c in 'CESTX'}

# R-1 chance baseline: coverage of own final code by other students' reply 8-grams (template excluded)
RG = {u: set(g for c in msg[(msg.u == u) & (msg.role == 'assistant')].content for g in grams(tok(c))) - TG for u in users}
above, corr, nullmed = [], [], []
for i, u in enumerate(users):
    F = R[u]['F']; null = np.array([cover(F, RG[v]).mean() if len(F) else 0 for v in users if v != u])
    above.append(U[i] > np.percentile(null, 99)); nullmed.append(np.median(null)); corr.append(U[i] - np.median(null))
corr = np.array(corr); above = np.array(above)
own_cov = np.array([cover(R[u]['F'], RG[u]).mean() for u in users])
out['chance'] = {'share_above_null99': float(above.mean()), 'share_above_ci': bci(np.mean, above.astype(float)),
                 'null_median_mean': float(np.mean(nullmed)), 'corrected_median': float(np.median(corr)), 'corrected_mean': float(corr.mean()),
                 'corrected_mean_ci': bci(np.mean, corr), 'own_cov_no_order_mean': float(own_cov.mean())}
# R-2 excluding reference-solution n-grams; R-3 copied strings
cgr = {u: set() for u in users}
for _, row in cp.iterrows():
    if row.userId in cgr:
        for st, t in json.loads(row.data): cgr[row.userId] |= set(grams(tok(st)))
def restricted(u, keep):
    r = R[u]; G = {g for g, c in r['cls'].items() if c == 'C' and keep(g)}; return cover(r['F'], G).mean() if len(r['F']) else 0
Ur = np.array([restricted(u, lambda g: g not in REF) for u in users]); Uc = np.array([restricted(u, lambda g, u=u: g in cgr[u]) for u in users])
for k, v in [('U_nonref', Ur), ('U_copied', Uc)]:
    out[k] = {'median': float(np.median(v)), 'mean': float(v.mean()), 'mean_ci': bci(np.mean, v), 'ge25': float((v >= .25).mean()), 'share_of_U': float(v.sum() / U.sum())}
out['U'] = {'median': float(np.median(U)), 'mean': float(U.mean())}

# R-4 segments with length controls
cl = cl[cl['Student ID'].isin(users)].sort_values(['Student ID', 'Timestamp (UTC)']).copy()
cl['seg'] = cl['Response-Use (detailed)'].notna().groupby(cl['Student ID']).cumsum()
FENCE = re.compile(r"```.*?```", re.S)
rows = []
for u, g in cl.groupby('Student ID'):
    g = g[g.seg > 0]; starts = g.groupby('seg')['Timestamp (UTC)'].min().sort_index(); r = R[u]
    for k, (sid, st) in enumerate(starts.items()):
        en = starts.iloc[k + 1] if k + 1 < len(starts) else np.inf; rr = g[g.seg == sid]
        rep = rr[rr.Role == 'Chatbot'].Message.astype(str); Gs = {x for t in rep for x in grams(tok(t))}
        up = sum(1 for x in Gs if r['cls'].get(x) == 'C' and st <= r['ted'].get(x, -1) < en)
        rows.append(dict(u=u, label=rr['Response-Use (detailed)'].dropna().iloc[0], up=up, nG=len(Gs),
                         rtok=sum(len(tok(t)) for t in rep), ctok=sum(len(tok(b)) for t in rep for b in FENCE.findall(t))))
sg = pd.DataFrame(rows); sg['upn'] = sg.up / sg.nG.clip(lower=1)
studs = sg.u.unique(); idx = {u: np.where(sg.u.values == u)[0] for u in studs}
def cb(fn):
    v = []
    for _ in range(NB):
        s_ = sg.iloc[np.concatenate([idx[u] for u in rng.choice(studs, len(studs))])]
        if s_.y.nunique() == 2: v.append(fn(s_))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def oof(Xc, y, groups, reps=20):
    P = np.zeros(len(y))
    for rep in range(reps):
        perm = {g: i for i, g in enumerate(np.random.default_rng(rep).permutation(np.unique(groups)))}
        gg = np.array([perm[g] for g in groups]); p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(Xc, y, gg):
            m = LogisticRegression(C=1.0, max_iter=1000).fit(Xc[tr], y[tr]); p[te] = m.predict_proba(Xc[te])[:, 1]
        P += p / reps
    return P
out['seg_n_students'] = int(sg.u.nunique()); out['seg_n'] = int(len(sg))
for nm, pos in [('copy', ['copy the response']), ('copy_or_writeover', ['copy the response', 'write over the response'])]:
    sg['y'] = sg.label.isin(pos).astype(int)
    base = np.c_[np.log1p(sg.rtok), np.log1p(sg.ctok)]; full = np.c_[base, np.log1p(sg.up)]
    sg['pb'] = oof(base, sg.y.values, sg.u.values); sg['pf'] = oof(full, sg.y.values, sg.u.values)
    o = {'n_pos': int(sg.y.sum())}
    for k in ['rtok', 'ctok', 'upn', 'up']: o[f'auc_{k}'] = [float(roc_auc_score(sg.y, sg[k])), cb(lambda s_, k=k: roc_auc_score(s_.y, s_[k]))]
    o['auc_base_oof'] = float(roc_auc_score(sg.y, sg.pb)); o['auc_full_oof'] = float(roc_auc_score(sg.y, sg.pf))
    o['gain'] = [o['auc_full_oof'] - o['auc_base_oof'], cb(lambda s_: roc_auc_score(s_.y, s_.pf) - roc_auc_score(s_.y, s_.pb))]
    out[f'seg_{nm}'] = o

# R-6 censored post-test
st = pd.DataFrame({'pre': ms.loc[users, 'Pre-test'].values, 'post': ms.loc[users, 'Post-test'].values, 'U': U})
z = (st.U - st.U.mean()) / st.U.std(); X = np.c_[np.ones(len(st)), st.pre, z]; y = st.post.values.astype(float); cens = y >= 10
def nll(th):
    b, s = th[:3], np.exp(th[3]); mu = X @ b
    return -(stats.norm.logpdf(y[~cens], mu[~cens], s).sum() + stats.norm.logsf(10, mu[cens], s).sum())
ols = np.linalg.lstsq(X, y, rcond=None)[0]; fit = optimize.minimize(nll, np.r_[ols, 0.0], method='BFGS')
se = np.sqrt(np.diag(fit.hess_inv)); bz, sz = fit.x[2], se[2]
out['tobit'] = {'zU': float(bz), 'ci': [float(bz - 1.96 * sz), float(bz + 1.96 * sz)], 'p': float(2 * stats.norm.sf(abs(bz / sz))), 'censored_share': float(cens.mean())}
from statsmodels.miscmodels.ordinal_model import OrderedModel
om = OrderedModel(y.astype(int), pd.DataFrame({'pre': st.pre, 'zU': z}), distr='logit').fit(method='bfgs', disp=0)
out['ordinal'] = {'or_zU': float(np.exp(om.params['zU'])), 'ci': [float(np.exp(v)) for v in om.conf_int().loc['zU']], 'p': float(om.pvalues['zU'])}
json.dump(out, open('results/r4_rs_stats.json', 'w'), indent=1); print(json.dumps(out, indent=1))
