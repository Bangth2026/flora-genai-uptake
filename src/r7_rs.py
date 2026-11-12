"""Plan R7, RelianceScope (PREREG_R7.md H2): agreement of segment uptake with each human response-use code.
Readers, attribution and segment construction are copied from src/r5_rs.py.
Usage: python -I src/r7_rs.py RS_DIR PROMPT_TXT -> results/r7_rs.json"""
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


rng = np.random.default_rng(2071)
o, U, corr, Ur, R, TG, REF = measures('alnum', 8)
sg = segments(R, LEX['alnum'], 8)
studs = sg.u.unique(); idx = {u: np.where(sg.u.values == u)[0] for u in studs}
B = [sg.iloc[np.concatenate([idx[u] for u in rng.choice(studs, len(studs))])] for _ in range(NB)]
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out = {'n_segments': int(len(sg)), 'n_students': int(len(studs)), 'check_auc_copy': float(roc_auc_score(sg.y, sg.up)), 'codes': {}}
for lab, n in sg.label.value_counts().items():
    if n < 8: continue
    yy = (sg.label == lab).astype(int)
    def stats_(s_):
        y_ = (s_.label == lab).astype(int)
        a = roc_auc_score(y_, s_.up) if y_.nunique() == 2 else np.nan
        g = s_[s_.label == lab]
        return (g.up > 0).mean() if len(g) else np.nan, g.up.mean() if len(g) else np.nan, a
    bs = np.array([stats_(s_) for s_ in B])
    g = sg[sg.label == lab]
    out['codes'][lab] = {'n': int(n), 'share_any_uptake': float((g.up > 0).mean()), 'share_ci': ci(bs[~np.isnan(bs[:, 0]), 0]),
                         'mean_uptake': float(g.up.mean()), 'mean_ci': ci(bs[~np.isnan(bs[:, 1]), 1]),
                         'auc_one_vs_rest': float(roc_auc_score(yy, sg.up)), 'auc_ci': ci(bs[~np.isnan(bs[:, 2]), 2])}
json.dump(out, open('results/r7_rs.json', 'w'), indent=1); print(json.dumps(out, indent=1))
