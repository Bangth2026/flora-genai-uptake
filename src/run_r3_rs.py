"""Plan R3, part R: provenance classes on the RelianceScope programming dataset.
Usage: python -I src/run_r3_rs.py RS_DIR -> results/r3_rs_stats.json, results/r3_rs_students.csv"""
import sys, json, re, warnings
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2030); NB = 2000; JUMP = 20
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, n: [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
L = lambda f: pd.DataFrame(list(json.load(open(f'{D}/logs/{f}')).values())[0])
msg, ed, conv = L('messages.json'), L('edit logs.json'), L('conversations.json')
cp = L('time and copy & paste logs.json'); cp = cp[cp.action == 'copied strings']
c2u = dict(zip(conv.id, conv.userId)); msg['u'] = msg.conversationId.map(c2u); ed['u'] = ed.conversationId.map(c2u)
ms = pd.read_csv(f'{D}/csv/measurement (by student).csv').set_index('Student ID')
cl = pd.read_csv(f'{D}/csv/chat logs (with classification).csv')
users = sorted(set(ms.index) & set(msg.u.dropna()) & set(ed.u.dropna()))

# editor stream: edit snapshots plus the code snapshot attached to each student message
snaps = [ed[['u', 'createdAt', 'code']]]
um = msg[(msg.role == 'user') & msg.snapshot.notna()].copy()
um['code'] = um.snapshot.map(lambda s: s.get('code') if isinstance(s, dict) else None); snaps.append(um[['u', 'createdAt', 'code']])
S = pd.concat(snaps).dropna().sort_values(['u', 'createdAt'])
first = ed.sort_values('createdAt').groupby('u').code.first()
template = Counter(first).most_common(1)[0][0]

def student(u, n=8, peer=None):
    x = msg[msg.u == u].sort_values('createdAt'); s = S[S.u == u]
    tbot, tpr = {}, {}
    for t, r, c in zip(x.createdAt, x.role, x.content):
        for g in grams(tok(c), n): (tbot if r == 'assistant' else tpr).setdefault(g, t)
    ted, mode, prev = {}, {}, 0
    for t, c in zip(s.createdAt, s.code):
        tk = tok(c); ch = abs(len(tk) - prev); prev = len(tk)
        for g in grams(tk, n):
            if g not in ted: ted[g] = t; mode[g] = 'T' if ch <= JUMP else 'X'
    F = tok(ed[ed.u == u].sort_values('createdAt').code.iloc[-1]); FG = grams(F, n)
    Tm = set(grams(tok(template), n)) | (peer or set())
    cls = {}
    for g in set(FG):
        if g in tbot and not (g in tpr and tpr[g] < tbot[g]) and g in ted and tbot[g] <= ted[g] and g not in Tm: cls[g] = 'C'
        elif g in tbot and ((g in tpr and tpr[g] < tbot[g]) or (g in ted and ted[g] < tbot[g])): cls[g] = 'E'
        elif g in Tm: cls[g] = 'S'
        else: cls[g] = mode.get(g, 'X')
    lab = np.array(['X'] * len(F), dtype=object); rank = {'C': 0, 'E': 1, 'S': 2, 'T': 3, 'X': 4}
    best = np.full(len(F), 9)
    for i, g in enumerate(FG):
        r = rank[cls[g]]
        best[i:i + n] = np.minimum(best[i:i + n], r)
    inv = {v: k for k, v in rank.items()}; lab = np.array([inv.get(b, 'X') for b in best])
    return dict(F=F, lab=lab, tbot=tbot, ted=ted, mode=mode, cls=cls)

R = {u: student(u) for u in users}
share = lambda u, c, RR=R: float((RR[u]['lab'] == c).mean()) if len(RR[u]['lab']) else 0.0
df = pd.DataFrame({k: [share(u, k) for u in users] for k in 'CESTX'}, index=users); df['ntok'] = [len(R[u]['F']) for u in users]
U = df.C.values
boot = lambda f, x: [float(np.percentile([f(rng.choice(x, len(x))) for _ in range(NB)], q)) for q in (2.5, 97.5)]
out = {'n_students': len(users), 'final_tokens_median': float(df.ntok.median()),
       'class_means': {k: float(df[k].mean()) for k in 'CESTX'},
       'U': {'median': float(np.median(U)), 'mean': float(U.mean()), 'ge25': float((U >= .25).mean()), 'ge50': float((U >= .5).mean()),
             'ge25_ci': boot(lambda v: (v >= .25).mean(), U), 'ge50_ci': boot(lambda v: (v >= .5).mean(), U), 'median_ci': boot(np.median, U)}}
for n in (5, 12):
    Rn = {u: student(u, n) for u in users}; Un = np.array([share(u, 'C', Rn) for u in users])
    out[f'U_n{n}'] = {'median': float(np.median(Un)), 'ge50': float((Un >= .5).mean()), 'spearman_with_n8': float(spearmanr(U, Un)[0])}
cnt = Counter(g for u in users for g in set(grams(R[u]['F'], 8)))
Rp = {}
for u in users:
    own = set(grams(R[u]['F'], 8)); Rp[u] = student(u, peer={g for g in own if cnt[g] - 1 >= 5})
Up = np.array([share(u, 'C', Rp) for u in users])
out['U_peer_filter_k5'] = {'median': float(np.median(Up)), 'mean': float(Up.mean()), 'ge50': float((Up >= .5).mean()),
                          'share_of_C_removed': float(1 - Up.sum() / U.sum()) if U.sum() else None}

# R-b copy events
cs, cov_c = [], []
for u in users:
    x = msg[(msg.u == u) & (msg.role == 'assistant')]
    for _, row in cp[cp.userId == u].iterrows():
        for st, t in json.loads(row.data):
            tk = tok(st)
            if not tk: continue
            prior = ' '.join(' '.join(tok(c)) for c in x[x.createdAt <= t].content)
            hit = any(g in R[u]['tbot'] and R[u]['tbot'][g] <= t for g in grams(tk, 8)) if len(tk) >= 8 else (' ' + ' '.join(tk) + ' ') in (' ' + prior + ' ')
            cs.append(dict(u=u, t=t, ntok=len(tk), from_bot=hit, tk=tk))
cs = pd.DataFrame(cs)
cg = {u: set(g for tk in cs[cs.u == u].tk for g in grams(tk, 8)) for u in users}
ctok, cin = 0, 0
for u in users:
    F, lab = R[u]['F'], R[u]['lab']; m = np.zeros(len(F), bool)
    for i, g in enumerate(grams(F, 8)):
        if g in cg[u]: m[i:i + 8] = True
    ctok += (lab == 'C').sum(); cin += ((lab == 'C') & m).sum()
out['copy'] = {'n_strings_nonempty': int(len(cs)), 'students_with_copies': int(cs.u.nunique()), 'share_from_bot': float(cs.from_bot.mean()),
               'share_from_bot_ge8tok': float(cs[cs.ntok >= 8].from_bot.mean()), 'n_ge8tok': int((cs.ntok >= 8).sum()),
               'share_C_tokens_in_copied': float(cin / ctok) if ctok else None}

# R-c human criterion (segments)
cl = cl[cl['Student ID'].isin(users)].sort_values(['Student ID', 'Timestamp (UTC)']).copy()
cl['seg'] = cl['Response-Use (detailed)'].notna().groupby(cl['Student ID']).cumsum()
segs = []
for u, g in cl.groupby('Student ID'):
    g = g[g.seg > 0]; starts = g.groupby('seg')['Timestamp (UTC)'].min().sort_index(); r = R[u]
    for k, (sid, st) in enumerate(starts.items()):
        en = starts.iloc[k + 1] if k + 1 < len(starts) else np.inf; rows = g[g.seg == sid]
        lab = rows['Response-Use (detailed)'].dropna().iloc[0]; rep = rows[rows.Role == 'Chatbot'].Message.astype(str)
        G = {gg for t in rep for gg in grams(tok(t), 8)}
        up = [gg for gg in G if gg in r['ted'] and st <= r['ted'][gg] < en and gg in r['tbot'] and r['tbot'][gg] <= r['ted'][gg]
              and not (gg in set(grams(tok(template), 8)))]
        segs.append(dict(u=u, label=lab, ru=rows['Response-Use'].dropna().iloc[0], hs=rows['Help-Seeking'].dropna().iloc[0] if rows['Help-Seeking'].notna().any() else None,
                         up=len(up), large=float(np.mean([r['mode'][gg] == 'X' for gg in up])) if up else np.nan))
sg = pd.DataFrame(segs); sg['pos'] = sg.label.isin(['copy the response', 'write over the response']).astype(int)
studs = sg.u.unique(); idx = {u: np.where(sg.u.values == u)[0] for u in studs}
def cb(fn):
    v = []
    for _ in range(NB):
        ii = np.concatenate([idx[u] for u in rng.choice(studs, len(studs))]); s_ = sg.iloc[ii]
        if s_.pos.nunique() == 2: v.append(fn(s_))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out['segments'] = {'n': int(len(sg)), 'n_pos': int(sg.pos.sum()), 'auc': float(roc_auc_score(sg.pos, sg.up)), 'auc_ci': cb(lambda s: roc_auc_score(s.pos, s.up)),
                   'any_uptake_by_code': {k: [float(v.mean()), int(len(v))] for k, v in (sg.up > 0).groupby(sg.label)},
                   'auc_copy_vs_other_nonpos': float(roc_auc_score(sg[sg.label != 'write over the response'].label == 'copy the response', sg[sg.label != 'write over the response'].up)),
                   'large_insertion_share': {k: [float(sg[(sg.label == k) & (sg.up > 0)].large.mean()), int(((sg.label == k) & (sg.up > 0)).sum())]
                                             for k in ['copy the response', 'write over the response']}}
# R-d student level
st = pd.DataFrame(index=users)
st['U'] = U; st['pos_share'] = sg.groupby('u').pos.mean(); st['passive_ru'] = sg.assign(p=sg.ru == 'Passive').groupby('u').p.mean()
st['passive_hs'] = sg.assign(p=sg.hs == 'Passive').groupby('u').p.mean(); st['perceived'] = ms.loc[users, 'Perceived reliance']
st['pre'] = ms.loc[users, 'Pre-test']; st['post'] = ms.loc[users, 'Post-test']
out['student'] = {}
for k in ['pos_share', 'passive_ru', 'passive_hs', 'perceived']:
    m = st[[k, 'U']].dropna(); r_ = spearmanr(m[k], m.U); out['student'][k] = {'rho': float(r_[0]), 'p': float(r_[1]), 'n': int(len(m))}
# R-e learning
z = (st.U - st.U.mean()) / st.U.std(); X = sm.add_constant(pd.DataFrame({'pre': st.pre, 'zU': z}))
f = sm.OLS(st.post, X).fit(cov_type='HC3'); ceil = (st.post == 10).astype(int)
lg = sm.Logit(ceil, X).fit(disp=0)
out['learning'] = {'ols_zU': float(f.params.zU), 'ci': [float(v) for v in f.conf_int().loc['zU']], 'p': float(f.pvalues.zU), 'ols_pre': float(f.params.pre),
                   'share_ceiling': float(ceil.mean()), 'logit_or_zU': float(np.exp(lg.params.zU)), 'logit_ci': [float(np.exp(v)) for v in lg.conf_int().loc['zU']],
                   'logit_p': float(lg.pvalues.zU), 'n': int(len(st))}
st.join(df).to_csv('results/r3_rs_students.csv'); json.dump(out, open('results/r3_rs_stats.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
