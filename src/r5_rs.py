"""Plan R5, RelianceScope (ANALYSIS_LOG_R5.md R1-R4): code-aware tokenisation, outcome models with bounded measures,
origin of copied strings, token-level projections and segment-local entry rule.
Usage: python -I src/r5_rs.py RS_DIR PROMPT_TXT -> results/r5_rs.json"""
import sys, json, re, warnings
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
warnings.filterwarnings('ignore')
D, PROMPT = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(2051); NB = 2000; J = 20
ALNUM = re.compile(r"[a-z0-9]+")
CODE = re.compile(r"[A-Za-z0-9_$@:\-]+|[^\sA-Za-z0-9_$@:\-]")
KW = set("""break case catch class const continue debugger default delete do else export extends finally for function if import in
instanceof let new return super switch this throw try typeof var void while with yield async await of true false null undefined
data methods computed watch mounted created template script style div input button ul ol li span p h1 h2 h3 form label
createApp mount Vue vue push filter splice trim length value id text key v-model v-for v-if v-else v-show v-on v-bind @click
@submit @keyup :key :class enter prevent""".split())
LEX = {'alnum': lambda s: ALNUM.findall(str(s).lower()),
       'code': lambda s: CODE.findall(str(s)),
       'code_norm': lambda s: [t if (t in KW or not re.match(r"[A-Za-z0-9_$@:\-]", t)) else ('LIT' if t[0].isdigit() else 'ID') for t in CODE.findall(str(s))]}
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
def cover(F, G, k):
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
ptxt = open(PROMPT).read(); REFTXT = ptxt[ptxt.index('<Answer Code>'):ptxt.index('</Answer Code>')]
FINAL = {u: ed[ed.u == u].sort_values('createdAt').code.iloc[-1] for u in users}
cl = cl[cl['Student ID'].isin(users)].sort_values(['Student ID', 'Timestamp (UTC)']).copy()
cl['seg'] = cl['Response-Use (detailed)'].notna().groupby(cl['Student ID']).cumsum()

def student(u, tok, n, TG, order='CESTX', mode_key='whole', proj='min'):
    x = msg[msg.u == u].sort_values('createdAt'); s = S[S.u == u]; tbot, tpr = {}, {}
    for t, r, c in zip(x.createdAt, x.role, x.content):
        for g in grams(tok(c), n): (tbot if r == 'assistant' else tpr).setdefault(g, t)
    F = tok(FINAL[u]); FG = grams(F, n); need = set(FG); ted, mw, msz, prev = {}, {}, {}, []
    for t, c in zip(s.createdAt, s.code):
        tk = tok(c); new = [g for g in set(grams(tk, n)) if g in need]
        if new:
            ops = SequenceMatcher(None, prev, tk, autojunk=False).get_opcodes(); segs = [(j1, j2) for op, i1, i2, j1, j2 in ops if op in ('insert', 'replace')]
            size = sum(j2 - j1 for j1, j2 in segs); pos = {}
            for k in range(len(tk) - n + 1):
                g = tuple(tk[k:k + n])
                if g in need and g not in pos: pos[g] = k
            for g in new:
                k = pos[g]; ov = [(min(k + n, j2) - max(k, j1), j2 - j1) for j1, j2 in segs]; ov = [o for o in ov if o[0] > 0]
                ted[g] = t; mw[g] = 'T' if size <= J else 'X'; msz[g] = 'T' if (max(ov)[1] if ov else 0) <= J else 'X'; need.discard(g)
        prev = tk
    mode = mw if mode_key == 'whole' else msz; cls = {}
    for g in set(FG):
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]
        if bot and not echo and g in ted and tbot[g] <= ted[g] and g not in TG: cls[g] = 'C'
        elif bot and (echo or (g in ted and ted[g] < tbot[g])): cls[g] = 'E'
        elif g in TG: cls[g] = 'S'
        else: cls[g] = mode.get(g, 'X')
    rank = {c: i for i, c in enumerate(order)}
    if proj == 'min':
        best = np.full(len(F), 99)
        for i, g in enumerate(FG): best[i:i + n] = np.minimum(best[i:i + n], rank[cls[g]])
        lab = np.array([order[b] if b < 99 else 'X' for b in best])
    else:
        votes = [Counter() for _ in F]
        for i, g in enumerate(FG):
            for j in range(i, i + n): votes[j][cls[g]] += 1
        lab = np.array([min(v, key=lambda c: (-v[c], rank[c])) if v else 'X' for v in votes])
    return dict(F=F, lab=lab, cls=cls, ted=ted)

def segments(R, tok, n):
    rows = []
    for u, g in cl.groupby('Student ID'):
        g = g[g.seg > 0]; starts = g.groupby('seg')['Timestamp (UTC)'].min().sort_index(); r = R[u]
        for k, (sid, st) in enumerate(starts.items()):
            en = starts.iloc[k + 1] if k + 1 < len(starts) else np.inf; rr = g[g.seg == sid]
            Gs = {x for t in rr[rr.Role == 'Chatbot'].Message.astype(str) for x in grams(tok(t), n)}
            rows.append(dict(u=u, label=rr['Response-Use (detailed)'].dropna().iloc[0], up=sum(1 for x in Gs if r['cls'].get(x) == 'C' and st <= r['ted'].get(x, -1) < en)))
    sg = pd.DataFrame(rows); sg['y'] = (sg.label == 'copy the response').astype(int); return sg

def cb_seg(sg, fn):
    studs = sg.u.unique(); idx = {u: np.where(sg.u.values == u)[0] for u in studs}; v = []
    for _ in range(NB):
        s_ = sg.iloc[np.concatenate([idx[u] for u in rng.choice(studs, len(studs))])]
        if s_.y.nunique() == 2: v.append(fn(s_))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
bci = lambda f, v: [float(np.percentile([f(rng.choice(v, len(v))) for _ in range(NB)], q)) for q in (2.5, 97.5)]

def measures(lex, n):
    tok = LEX[lex]; TG = set(grams(tok(template), n)); REF = set(grams(tok(REFTXT), n))
    R = {u: student(u, tok, n, TG) for u in users}; U = np.array([(R[u]['lab'] == 'C').mean() for u in users])
    RG = {u: set(g for c in msg[(msg.u == u) & (msg.role == 'assistant')].content for g in grams(tok(c), n)) - TG for u in users}
    nullmed = np.array([np.median([cover(R[u]['F'], RG[v], n).mean() for v in users if v != u]) for u in users]); corr = U - nullmed
    Ur = np.array([cover(R[u]['F'], {g for g, c in R[u]['cls'].items() if c == 'C' and g not in REF}, n).mean() for u in users])
    sg = segments(R, tok, n)
    o = {'U_mean': float(U.mean()), 'U_median': float(np.median(U)), 'null_median_mean': float(nullmed.mean()),
         'corrected_mean': float(corr.mean()), 'corrected_mean_ci': bci(np.mean, corr), 'nonref_mean': float(Ur.mean()), 'nonref_mean_ci': bci(np.mean, Ur),
         'auc_copy_up': [float(roc_auc_score(sg.y, sg.up)), cb_seg(sg, lambda s_: roc_auc_score(s_.y, s_.up))],
         'mean_tokens_final': float(np.mean([len(R[u]['F']) for u in users]))}
    return o, U, corr, Ur, R, TG, REF

out = {'n_students': len(users)}
base = None
for lex in ['alnum', 'code', 'code_norm']:
    for n in (8, 12):
        o, U, corr, Ur, R, TG, REF = measures(lex, n); out[f'lex_{lex}_n{n}'] = o; print(lex, n, o, flush=True)
        if lex == 'alnum' and n == 8: base = (U, corr, Ur, R, TG, REF)
U, corr, Ur, R, TG, REF = base
# R4 projections and segment-local entry (alnum, n = 8)
tok = LEX['alnum']
for key, kw in [('EC', dict(order='ECSTX')), ('MAJ', dict(proj='maj')), ('SEG', dict(mode_key='seg'))]:
    RR = {u: student(u, tok, 8, TG, **kw) for u in users}
    sh = {c: float(np.mean([(RR[u]['lab'] == c).mean() for u in users])) for c in 'CESTX'}
    sh['tokens_changed_share'] = sum(int((RR[u]['lab'] != R[u]['lab']).sum()) for u in users) / sum(len(R[u]['F']) for u in users)
    out[f'classes_{key}'] = sh
out['classes_R4'] = {c: float(np.mean([(R[u]['lab'] == c).mean() for u in users])) for c in 'CESTX'}
# R2 outcome models
refcov = np.array([len(set(grams(R[u]['F'], 8)) & REF) / max(1, len(REF)) for u in users])
st = pd.DataFrame({'pre': ms.loc[users, 'Pre-test'].values.astype(float), 'post': ms.loc[users, 'Post-test'].values.astype(float),
                   'refcov': refcov, 'minutes': ms.loc[users, 'Activity (min.)'].values.astype(float)})
y = st.post.values; cens = y >= 10
def tobit(X):
    def nll(th):
        b, s = th[:-1], np.exp(th[-1]); mu = X @ b
        return -(stats.norm.logpdf(y[~cens], mu[~cens], s).sum() + stats.norm.logsf(10, mu[cens], s).sum())
    ols = np.linalg.lstsq(X, y, rcond=None)[0]; f = optimize.minimize(nll, np.r_[ols, 0.0], method='BFGS'); se = np.sqrt(np.diag(f.hess_inv))
    return float(f.x[-2 if False else 2]), float(se[2])
res = {}
for pname, v in [('U', U), ('U_corrected', corr), ('U_nonref', Ur)]:
    z = (v - v.mean()) / v.std()
    for adj in ['pre', 'pre+progress']:
        cov = ['pre'] if adj == 'pre' else ['pre', 'refcov', 'minutes']
        Xd = pd.DataFrame({'z': z, **{c: st[c].values for c in cov}})
        X = np.c_[np.ones(len(st)), Xd[['pre', 'z'] + cov[1:]].values]
        b, se = tobit(X)
        f = sm.OLS(y, sm.add_constant(Xd)).fit(cov_type='HC3')
        om = OrderedModel(y.astype(int), Xd, distr='logit').fit(method='bfgs', disp=0)
        res[f'{pname}|{adj}'] = {'tobit': [b, [b - 1.96 * se, b + 1.96 * se], float(2 * stats.norm.sf(abs(b / se)))],
                                 'ols': [float(f.params['z']), [float(c) for c in f.conf_int().loc['z']], float(f.pvalues['z'])],
                                 'ordinal_or': [float(np.exp(om.params['z'])), [float(np.exp(c)) for c in om.conf_int().loc['z']], float(om.pvalues['z'])]}
out['outcomes'] = res
out['progress'] = {'refcov_mean': float(refcov.mean()), 'refcov_rho_U': float(stats.spearmanr(refcov, U)[0]), 'refcov_rho_post': float(stats.spearmanr(refcov, y)[0]),
                   'corr_rho_nonref': float(stats.spearmanr(corr, Ur)[0])}
# R3 origin of copied strings
rep = msg[msg.role == 'assistant'][['u', 'createdAt', 'content']]
cat = Counter()
for _, row in cp.iterrows():
    u = row.userId
    for s_, t in json.loads(row.data):
        tk = tok(s_)
        if not tk: continue
        q = ' ' + ' '.join(tk) + ' '
        inrep = any(q in ' ' + ' '.join(tok(c)) + ' ' for c in rep[(rep.u == u) & (rep.createdAt < t)].content)
        incode = any(q in ' ' + ' '.join(tok(c)) + ' ' for c in S[(S.u == u) & (S.createdAt < t)].code)
        L_ = 'ge8' if len(tk) >= 8 else 'lt8'
        cat[(L_, 'both' if inrep and incode else 'reply' if inrep else 'code' if incode else 'neither')] += 1
out['copied_strings'] = {f'{a}|{b}': v for (a, b), v in sorted(cat.items())}
json.dump(out, open('results/r5_rs.json', 'w'), indent=1); print(json.dumps(out, indent=1))
