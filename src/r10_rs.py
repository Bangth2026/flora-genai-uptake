"""Analysis log R10, RelianceScope: K-1 attribution of intermediate snapshots, K-2 effect of each parameter choice,
K-3 label-noise ceiling over q, W-2 winnowing and string tiling at student and segment level, W-3 contribution ratio.
Readers and attribution are copied from src/r8_rs.py (primary: n = 8, tau = 20, alphanumeric tokens).
Usage: python -I src/r10_rs.py RS_DIR PROMPT_TXT -> results/r10_rs.json"""
import sys, os, json, re, warnings
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demo'))
from baselines import winnow_mark, tile_mark
warnings.filterwarnings('ignore')
D, PROMPT = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(2101); NB = 2000
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
CP, WO = 'copy the response', 'write over the response'

def dialogue(u, tok, n):
    x = msg[msg.u == u].sort_values('createdAt'); tbot, tpr = {}, {}
    for t, r, c in zip(x.createdAt, x.role, x.content):
        for g in grams(tok(c), n): (tbot if r == 'assistant' else tpr).setdefault(g, t)
    return tbot, tpr

def attribute(u, tok, n, TG, art=None, upto=None, J=20, delta=0, order='CESTX', proj='min'):
    """Primary RelianceScope attribution (src/r8_rs.py, mode 'whole'); artefact = final code unless art is given;
    snapshots after time upto are ignored."""
    tbot, tpr = dialogue(u, tok, n); s = S[S.u == u]
    if upto is not None: s = s[s.createdAt <= upto]
    F = tok(FINAL[u] if art is None else art); FG = grams(F, n); need = set(FG); ted, mw, prev = {}, {}, []
    for t, c in zip(s.createdAt, s.code):
        tk = tok(c); new = [g for g in set(grams(tk, n)) if g in need]
        if new:
            ops = SequenceMatcher(None, prev, tk, autojunk=False).get_opcodes(); size = sum(j2 - j1 for op, i1, i2, j1, j2 in ops if op in ('insert', 'replace'))
            for g in new: ted[g] = t; mw[g] = 'T' if size <= J else 'X'; need.discard(g)
        prev = tk
    cls = {}
    for g in set(FG):
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]
        if bot and not echo and g in ted and tbot[g] <= ted[g] + delta and g not in TG: cls[g] = 'C'
        elif bot and (echo or (g in ted and ted[g] < tbot[g])): cls[g] = 'E'
        elif g in TG: cls[g] = 'S'
        else: cls[g] = mw.get(g, 'X')
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

def seg_table():
    rows = []
    for u, g in cl.groupby('Student ID'):
        g = g[g.seg > 0]; starts = g.groupby('seg')['Timestamp (UTC)'].min().sort_index()
        for k, (sid, st) in enumerate(starts.items()):
            en = starts.iloc[k + 1] if k + 1 < len(starts) else np.inf; rr = g[g.seg == sid]
            rows.append(dict(u=u, sid=sid, st=st, en=en, last_reply=rr[rr.Role == 'Chatbot']['Timestamp (UTC)'].max(),
                             label=rr['Response-Use (detailed)'].dropna().iloc[0], texts=rr[rr.Role == 'Chatbot'].Message.astype(str).tolist()))
    return pd.DataFrame(rows)
SG = seg_table()

def seg_auc(sg, col):
    o = {}
    studs = sg.u.unique(); idx = {u: np.where(sg.u.values == u)[0] for u in studs}
    for lab in (CP, WO):
        y = (sg.label == lab).astype(int); v = []
        for _ in range(NB):
            s_ = sg.iloc[np.concatenate([idx[u] for u in rng.choice(studs, len(studs))])]; yy = (s_.label == lab).astype(int)
            if yy.nunique() == 2: v.append(roc_auc_score(yy, s_[col]))
        o[lab] = {'auc': float(roc_auc_score(y, sg[col])), 'ci': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                  'share_any': float((sg.loc[sg.label == lab, col] > 0).mean())}
    return o

def seg_uptake_final(R, tok, n):
    return np.array([sum(1 for x in {x for t in r.texts for x in grams(tok(t), n)} if R[r.u]['cls'].get(x) == 'C' and r.st <= R[r.u]['ted'].get(x, -1) < r.en)
                     for r in SG.itertuples()])

def snapshot_at(u, t):
    s = S[(S.u == u) & (S.createdAt <= t)]
    return s.code.iloc[-1] if len(s) else ''

out = {'n_students': len(users), 'n_segments': int(len(SG))}
# ---------------- K-1 intermediate snapshots
K1 = {}
for lex, n in (('alnum', 8), ('alnum', 5), ('code', 8), ('code', 5)):
    tok = LEX[lex]; TG = set(grams(tok(template), n))
    R = {u: attribute(u, tok, n, TG) for u in users}; sg = SG.copy(); sg['up_final'] = seg_uptake_final(R, tok, n)
    for name, when in (('segment_end', lambda r: r.en - 1 if np.isfinite(r.en) else np.inf), ('reply_plus_2min', lambda r: r.last_reply + 120000), ('reply_plus_5min', lambda r: r.last_reply + 300000)):
        vals = []
        for r in sg.itertuples():
            tt = when(r); art = snapshot_at(r.u, tt) if np.isfinite(tt) else FINAL[r.u]
            a = attribute(r.u, tok, n, TG, art=art, upto=tt if np.isfinite(tt) else None)
            Gs = {x for t in r.texts for x in grams(tok(t), n)}
            vals.append(sum(1 for x in Gs if a['cls'].get(x) == 'C' and r.st <= a['ted'].get(x, -1) < min(r.en, tt + 1)))
        sg['up_' + name] = vals
    K1[f'{lex}_n{n}'] = {c: seg_auc(sg, c) for c in ('up_final', 'up_segment_end', 'up_reply_plus_2min', 'up_reply_plus_5min')}
    K1[f'{lex}_n{n}']['writeover_share_uptake_final_vs_segment_end'] = [float((sg.loc[sg.label == WO, 'up_final'] > 0).mean()), float((sg.loc[sg.label == WO, 'up_segment_end'] > 0).mean())]
    print(lex, n, {c: (round(v[CP]['auc'], 3), round(v[WO]['auc'], 3)) for c, v in K1[f'{lex}_n{n}'].items() if isinstance(v, dict)}, flush=True)
out['K1_intermediate_snapshots'] = K1

# ---------------- K-2 parameter table
tokA = LEX['alnum']
pre = ms.loc[users, 'Pre-test'].values.astype(float); post = ms.loc[users, 'Post-test'].values.astype(float)
def measures(R, tok, n):
    TG = set(grams(tok(template), n)); U = np.array([(R[u]['lab'] == 'C').mean() for u in users])
    RG = {u: set(g for c in msg[(msg.u == u) & (msg.role == 'assistant')].content for g in grams(tok(c), n)) - TG for u in users}
    nullmed = np.array([np.median([cover(R[u]['F'], RG[v], n).mean() for v in users if v != u]) for u in users])
    sg = SG.copy(); sg['up'] = seg_uptake_final(R, tok, n)
    z = (U - U.mean()) / U.std(); f = sm.OLS(post, sm.add_constant(np.c_[pre, z])).fit(cov_type='HC3')
    return {'U_mean': float(U.mean()), 'U_median': float(np.median(U)), 'corrected_mean': float((U - nullmed).mean()), 'ge50': float((U >= .5).mean()),
            'auc_copy': float(roc_auc_score((sg.label == CP).astype(int), sg.up)), 'auc_writeover': float(roc_auc_score((sg.label == WO).astype(int), sg.up)),
            'post_slope_per_sd': float(f.params[2]), 'post_slope_ci': [float(c) for c in f.conf_int()[2]]}
K2 = {}
SPECS = [('primary', dict(lex='alnum', n=8)), ('n5', dict(lex='alnum', n=5)), ('n12', dict(lex='alnum', n=12)),
         ('tau5', dict(lex='alnum', n=8, J=5)), ('tau50', dict(lex='alnum', n=8, J=50)), ('delta_plus60s', dict(lex='alnum', n=8, delta=60000)),
         ('E_before_C', dict(lex='alnum', n=8, order='ECSTX')), ('majority', dict(lex='alnum', n=8, proj='maj')),
         ('code_lexer', dict(lex='code', n=8)), ('code_lexer_n5', dict(lex='code', n=5)), ('code_norm', dict(lex='code_norm', n=8))]
for name, sp in SPECS:
    tok = LEX[sp['lex']]; n = sp['n']; TG = set(grams(tok(template), n))
    R = {u: attribute(u, tok, n, TG, J=sp.get('J', 20), delta=sp.get('delta', 0), order=sp.get('order', 'CESTX'), proj=sp.get('proj', 'min')) for u in users}
    K2[name] = measures(R, tok, n); print(name, K2[name], flush=True)
    if name == 'primary': RP = R
out['K2_parameters'] = K2

# ---------------- K-3 label-noise ceiling
yt = np.r_[np.ones(119), np.zeros(427 - 119)].astype(int); K3 = {}
for q in np.round(np.arange(0, .201, .02), 2):
    a = []
    for _ in range(10000 if q > 0 else 1):
        flip = rng.random(len(yt)) < q; yo = np.where(flip, 1 - yt, yt)
        if 0 < yo.sum() < len(yo): a.append(roc_auc_score(yo, yt + rng.random(len(yt)) * 1e-9))
    K3[str(q)] = {'mean_auc': float(np.mean(a)), 'p2.5': float(np.percentile(a, 2.5)), 'p97.5': float(np.percentile(a, 97.5))}
out['K3_noise_ceiling'] = K3

# ---------------- W-2 winnowing and string tiling; W-3 contribution ratio
tok = tokA; TG = set(grams(tok(template), 8)); W2 = {}
REP = {u: [tok(c) for c in msg[(msg.u == u) & (msg.role == 'assistant')].content] for u in users}
U = np.array([(RP[u]['lab'] == 'C').mean() for u in users]); per = (ms.loc[users, 'Perceived reliance'].values.astype(float))
for name, fn in (('winnowing', lambda F, srcs: winnow_mark(F, srcs, 8, 4)), ('string_tiling', lambda F, srcs: tile_mark(F, srcs, 8))):
    own, own_nt, chance, cls_share = [], [], [], Counter()
    for u in users:
        F = RP[u]['F']; m = np.array(fn(F, REP[u])); tm = cover(F, TG, 8)
        own.append(m.mean()); own_nt.append((m & ~tm).mean())
        for c in 'CESTX': cls_share[c] += int((m & (RP[u]['lab'] == c)).sum())
        others = rng.choice([v for v in users if v != u], 20, replace=False); chance.append(np.median([np.mean(fn(F, REP[v])) for v in others]))
    own, own_nt, chance = map(np.array, (own, own_nt, chance)); tot = sum(cls_share.values())
    sg = SG.copy(); sg['score'] = [int(np.sum(fn(RP[r.u]['F'], [tok(t) for t in r.texts]))) for r in sg.itertuples()]
    W2[name] = {'mean_share': float(own.mean()), 'median_share': float(np.median(own)), 'mean_share_template_removed': float(own_nt.mean()),
                'ge50': float((own >= .5).mean()), 'chance_median_mean': float(chance.mean()), 'corrected_mean': float((own - chance).mean()),
                'spearman_with_Ustar': float(pd.Series(own).corr(pd.Series(U), method='spearman')), 'spearman_with_perceived_reliance': float(pd.Series(own).corr(pd.Series(per), method='spearman')),
                'marked_tokens_by_class': {c: cls_share[c] / tot for c in 'CESTX'}, 'segment_agreement': seg_auc(sg, 'score')}
    print(name, W2[name], flush=True)
sg = SG.copy(); sg['up'] = seg_uptake_final(RP, tok, 8); W2['segment_uptake_primary'] = seg_auc(sg, 'up')
W2['Ustar_spearman_with_perceived_reliance'] = float(pd.Series(U).corr(pd.Series(per), method='spearman'))
Tsh = np.array([(RP[u]['lab'] == 'T').mean() for u in users])
W2['contribution_ratio_T'] = {'mean_T': float(Tsh.mean()), 'median_T': float(np.median(Tsh)), 'spearman_T_with_1_minus_U': float(pd.Series(Tsh).corr(pd.Series(1 - U), method='spearman')),
                              'spearman_T_with_perceived_reliance': float(pd.Series(Tsh).corr(pd.Series(per), method='spearman'))}
out['W2_W3_baselines'] = W2
json.dump(out, open('results/r10_rs.json', 'w'), indent=1, default=float)
