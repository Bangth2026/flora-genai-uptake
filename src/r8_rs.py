"""Plan R8, RelianceScope (ANALYSIS_LOG_R8.md K1-K4): segments coded "write over the response" under other n, lexers,
windows and a label-free count; order of the E and S rules; agreement ceiling under label noise; post-test models
with pre-activity covariates instead of task progress. Readers and attribution are copied from src/r7_rs.py.
Usage: python -I src/r8_rs.py RS_DIR PROMPT_TXT -> results/r8_rs.json"""
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



from statsmodels.tools.numdiff import approx_hess
rng = np.random.default_rng(2081)
WO, CP = 'write over the response', 'copy the response'

def segments2(R, tok, n, window='next'):
    """Segment uptake with three windows; also label-free count of reply n-grams in the final code."""
    rows = []
    for u, g in cl.groupby('Student ID'):
        g = g[g.seg > 0]; starts = g.groupby('seg')['Timestamp (UTC)'].min().sort_index(); r = R[u]; FGs = set(grams(r['F'], n))
        for k, (sid, st) in enumerate(starts.items()):
            if window == 'next': en = starts.iloc[k + 1] if k + 1 < len(starts) else np.inf
            elif window == 'next2': en = starts.iloc[k + 2] if k + 2 < len(starts) else np.inf
            else: en = np.inf
            rr = g[g.seg == sid]; texts = rr[rr.Role == 'Chatbot'].Message.astype(str)
            Gs = {x for t in texts for x in grams(tok(t), n)}
            rows.append(dict(u=u, label=rr['Response-Use (detailed)'].dropna().iloc[0], st=st, en=en,
                             up=sum(1 for x in Gs if r['cls'].get(x) == 'C' and st <= r['ted'].get(x, -1) < en),
                             lf=sum(1 for x in Gs if x in FGs), codeblock=int(any('```' in t for t in texts)), Gs=Gs))
    return pd.DataFrame(rows)

def agree(sg, col):
    o = {}
    for lab in (CP, WO):
        y = (sg.label == lab).astype(int); g = sg[sg.label == lab]
        studs = sg.u.unique(); idx = {u: np.where(sg.u.values == u)[0] for u in studs}; v = []
        for _ in range(NB):
            s_ = sg.iloc[np.concatenate([idx[u] for u in rng.choice(studs, len(studs))])]; yy = (s_.label == lab).astype(int)
            if yy.nunique() == 2: v.append(roc_auc_score(yy, s_[col]))
        o[lab] = {'n': int(len(g)), 'share_any': float((g[col] > 0).mean()), 'auc_one_vs_rest': float(roc_auc_score(y, sg[col])),
                  'auc_ci': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    return o

out = {'n_students': len(users)}
K1 = {}; base = None
for lex, n in [('alnum', 5), ('alnum', 8), ('alnum', 12), ('code', 8), ('code', 5), ('code_norm', 8), ('code_norm', 5)]:
    tok = LEX[lex]; TG = set(grams(tok(template), n)); R = {u: student(u, tok, n, TG) for u in users}
    sg = segments2(R, tok, n); K1[f'{lex}_n{n}'] = agree(sg, 'up'); print(lex, n, K1[f'{lex}_n{n}'], flush=True)
    if lex == 'alnum' and n == 8:
        base = (R, TG, sg)
        for w_ in ('next2', 'open'):
            K1[f'alnum_n8_window_{w_}'] = agree(segments2(R, tok, 8, w_), 'up')
        K1['alnum_n8_label_free'] = agree(sg, 'lf')
    if lex == 'alnum' and n == 5: R5_, sg5 = R, sg
out['K1'] = K1
R, TG, sg = base
check = roc_auc_score((sg.label == CP).astype(int), sg.up); out['check_auc_copy_n8'] = float(check)
# write-over segments without uptake at n = 8
wo0 = sg[(sg.label == WO) & (sg.up == 0)]
# any reply 5-gram entering the editor within the segment window (alnum, n = 5 attribution)
sg5i = sg5.set_index(['u', 'st'])
def any5(row):
    r = R5_[row.u]; G5 = {x for x in sg5i.loc[(row.u, row.st)].Gs} if (row.u, row.st) in sg5i.index else set()
    return any(row.st <= r['ted'].get(x, -1) < row.en for x in G5)
out['K1_writeover_no_uptake'] = {'n': int(len(wo0)), 'n_writeover': int((sg.label == WO).sum()),
                                 'share_reply_has_code_block': float(wo0.codeblock.mean()),
                                 'share_any_reply_8gram_in_final_code': float((wo0.lf > 0).mean()),
                                 'share_any_reply_5gram_enters_editor_in_window': float(np.mean([any5(r_) for r_ in wo0.itertuples()])),
                                 'writeover_with_uptake_share_code_block': float(sg[(sg.label == WO) & (sg.up > 0)].codeblock.mean())}
# K2 order of E and S rules (template)
tok = LEX['alnum']; n = 8; ch = 0; tot = 0; grams_moved = 0; grams_S = 0; sh_new = {c: [] for c in 'CESTX'}
for u in users:
    x = msg[msg.u == u]; tbot, tpr = {}, {}
    for t, r_, c in zip(x.createdAt, x.role, x.content):
        for g in grams(tok(c), n): (tbot if r_ == 'assistant' else tpr).setdefault(g, t)
    cls = dict(R[u]['cls']); ted = R[u]['ted']
    for g, c in list(cls.items()):
        if c == 'S':
            grams_S += 1
            if g in tbot and g in ted and ted[g] < tbot[g]: cls[g] = 'E'; grams_moved += 1
    F = R[u]['F']; rank = {c: i for i, c in enumerate('CESTX')}; best = np.full(len(F), 99)
    for i, g in enumerate(grams(F, n)): best[i:i + n] = np.minimum(best[i:i + n], rank[cls[g]])
    lab = np.array(['CESTX'[b] if b < 99 else 'X' for b in best]); ch += int((lab != R[u]['lab']).sum()); tot += len(F)
    for c in 'CESTX': sh_new[c].append((lab == c).mean())
out['K2_E_before_S'] = {'S_grams': grams_S, 'S_grams_editor_before_reply': grams_moved, 'share_of_S_grams': grams_moved / max(1, grams_S),
                        'tokens_changed_share': ch / tot, 'class_means_E_first': {c: float(np.mean(v)) for c, v in sh_new.items()},
                        'class_means_R4': {c: float(np.mean([(R[u]['lab'] == c).mean() for u in users])) for c in 'CESTX'}}
# K3 agreement ceiling under label noise
yt = np.r_[np.ones(119), np.zeros(427 - 119)].astype(int); score_ = yt.astype(float); K3 = {}
for q in (0.05, 0.10, 0.15, 0.17):
    a = []
    for _ in range(10000):
        flip = rng.random(len(yt)) < q; yo = np.where(flip, 1 - yt, yt)
        if 0 < yo.sum() < len(yo): a.append(roc_auc_score(yo, score_ + rng.random(len(yt)) * 1e-9))
    K3[str(q)] = {'mean_auc': float(np.mean(a)), 'p2.5': float(np.percentile(a, 2.5)), 'p97.5': float(np.percentile(a, 97.5))}
out['K3_noise_ceiling'] = K3
# K4 post-test models with pre-activity covariates
REF = set(grams(tok(REFTXT), 8)); U = np.array([(R[u]['lab'] == 'C').mean() for u in users])
RG = {u: set(g for c in msg[(msg.u == u) & (msg.role == 'assistant')].content for g in grams(tok(c), 8)) - TG for u in users}
nullmed = np.array([np.median([cover(R[u]['F'], RG[v], 8).mean() for v in users if v != u]) for u in users]); corr = U - nullmed
Ur = np.array([cover(R[u]['F'], {g for g, c in R[u]['cls'].items() if c == 'C' and g not in REF}, 8).mean() for u in users])
out['check_means'] = {'U': float(U.mean()), 'corr': float(corr.mean()), 'nonref': float(Ur.mean())}
QV = ['Self-efficacy', 'Intrinsic motivation', 'Extrinsic motivation', 'Metacognition']
st = pd.DataFrame({'pre': ms.loc[users, 'Pre-test'].values.astype(float), 'post': ms.loc[users, 'Post-test'].values.astype(float),
                   **{f'q{i}': ms.loc[users, v].values.astype(float) for i, v in enumerate(QV)}})
y = st.post.values; cens = y >= 10
def tobit(X, j):
    def nll(th):
        b, s = th[:-1], np.exp(th[-1]); mu = X @ b
        return -(stats.norm.logpdf(y[~cens], mu[~cens], s).sum() + stats.norm.logsf(10, mu[cens], s).sum())
    ols = np.linalg.lstsq(X, y, rcond=None)[0]; f = optimize.minimize(nll, np.r_[ols, 0.0], method='BFGS')
    se_b = np.sqrt(np.diag(f.hess_inv)); se_n = np.sqrt(np.diag(np.linalg.inv(approx_hess(f.x, nll))))  # numerical Hessian; BFGS approximation kept for comparison
    return float(f.x[j]), float(se_n[j]), float(se_b[j])
K4 = {'questionnaire_missing': int(st[[f'q{i}' for i in range(4)]].isna().sum().sum())}
for pname, v in [('U', U), ('U_corrected', corr), ('U_nonref', Ur)]:
    z = (v - v.mean()) / v.std()
    for adj, cov in [('pre', ['pre']), ('pre+questionnaire', ['pre', 'q0', 'q1', 'q2', 'q3'])]:
        Xd = pd.DataFrame({'z': z, **{c: st[c].values for c in cov}})
        f = sm.OLS(y, sm.add_constant(Xd)).fit(cov_type='HC3')
        b, se, se_bfgs = tobit(np.c_[np.ones(len(st)), Xd.values], 1)
        K4[f'{pname}|{adj}'] = {'ols': [float(f.params['z']), [float(c) for c in f.conf_int().loc['z']], float(f.pvalues['z'])],
                                'tobit': [b, [b - 1.96 * se, b + 1.96 * se], float(2 * stats.norm.sf(abs(b / se)))], 'tobit_se_numerical': se, 'tobit_se_bfgs': se_bfgs}
K4['rho_questionnaire_U'] = {v: float(stats.spearmanr(st[f'q{i}'], U)[0]) for i, v in enumerate(QV)}
out['K4_outcomes'] = K4
json.dump(out, open('results/r8_rs.json', 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'K1'}, indent=1))
