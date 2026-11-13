"""Analysis log R12, K-4 (RelianceScope): the token entry rule, which is primary for FLoRA, applied to the code data,
and the composition of large insertions, to show why the unresolved class is almost empty.
Readers and the n-gram attribution are those of src/r8_rs.py (primary: n = 8, tau = 20, alphanumeric tokens).
Usage: python -I src/r12_rs.py RS_DIR PROMPT_TXT -> results/r12_rs.json"""
import sys, json, re
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
D, PROMPT = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(2140); NB = 2000
tok = lambda s: re.findall(r"[a-z0-9]+", str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
L = lambda f: pd.DataFrame(list(json.load(open(f'{D}/logs/{f}')).values())[0])
msg, ed, conv = L('messages.json'), L('edit logs.json'), L('conversations.json')
c2u = dict(zip(conv.id, conv.userId)); msg['u'] = msg.conversationId.map(c2u); ed['u'] = ed.conversationId.map(c2u)
ms = pd.read_csv(f'{D}/csv/measurement (by student).csv').set_index('Student ID')
users = sorted(set(ms.index) & set(msg.u.dropna()) & set(ed.u.dropna()))
um = msg[(msg.role == 'user') & msg.snapshot.notna()].copy()
um['code'] = um.snapshot.map(lambda s: s.get('code') if isinstance(s, dict) else None)
S = pd.concat([ed[['u', 'createdAt', 'code']].assign(src='editor'), um[['u', 'createdAt', 'code']].assign(src='message')]).dropna(subset=['code']).sort_values(['u', 'createdAt'])
template = Counter(ed.sort_values('createdAt').groupby('u').code.first()).most_common(1)[0][0]
FINAL = {u: ed[ed.u == u].sort_values('createdAt').code.iloc[-1] for u in users}
N = 8; TG = set(grams(tok(template), N))


def propagate(seq, tau):
    """Token entry rule: entry size of every token of the last snapshot."""
    prev, size, kin = [], [], []
    for k, cur in enumerate(seq):
        if not prev:
            size = [len(cur)] * len(cur); kin = [k] * len(cur); prev = cur; continue
        m = min(len(prev), len(cur)); pre = 0
        while pre < m and prev[pre] == cur[pre]: pre += 1
        suf = 0
        while suf < m - pre and prev[-1 - suf] == cur[-1 - suf]: suf += 1
        a, b = prev[pre:len(prev) - suf], cur[pre:len(cur) - suf]
        if not a and not b: prev = cur; continue
        mid = [None] * len(b); midk = [None] * len(b)
        ops = [('insert', 0, 0, 0, len(b))] if not a else SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        for op, i1, i2, j1, j2 in ops:
            for q in range(j2 - j1):
                mid[j1 + q] = size[pre + i1 + q] if op == 'equal' else (j2 - j1)
                midk[j1 + q] = kin[pre + i1 + q] if op == 'equal' else k
        size = size[:pre] + mid + (size[len(prev) - suf:] if suf else [])
        kin = kin[:pre] + midk + (kin[len(prev) - suf:] if suf else []); prev = cur
    return prev, size, kin


def student(u):
    x = msg[msg.u == u].sort_values('createdAt'); s = S[S.u == u]; tbot, tpr = {}, {}
    for t, r, c in zip(x.createdAt, x.role, x.content):
        for g in grams(tok(c), N): (tbot if r == 'assistant' else tpr).setdefault(g, t)
    F = tok(FINAL[u]); FG = grams(F, N); need = set(FG); ted, mw, prev = {}, {}, []
    seq = []
    for t, c in zip(s.createdAt, s.code):
        tk = tok(c); seq.append(tk); new = [g for g in set(grams(tk, N)) if g in need]
        if new:
            size = sum(j2 - j1 for op, i1, i2, j1, j2 in SequenceMatcher(None, prev, tk, autojunk=False).get_opcodes() if op in ('insert', 'replace'))
            for g in new: ted[g] = t; mw[g] = size; need.discard(g)
        prev = tk
    cls = {}
    for g in set(FG):
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]
        if bot and not echo and g in ted and tbot[g] <= ted[g] and g not in TG: cls[g] = 'C'
        elif bot and (echo or (g in ted and ted[g] < tbot[g])): cls[g] = 'E'
        elif g in TG: cls[g] = 'S'
        else: cls[g] = 'U'
    rank = {c: i for i, c in enumerate('CESU')}; best = np.full(len(F), 9)
    for i, g in enumerate(FG): best[i:i + N] = np.minimum(best[i:i + N], rank[cls[g]])
    ces = np.array(['CESU'[b] if b < 9 else 'U' for b in best])
    # n-gram entry size of the earliest-entering n-gram covering each token (as in the primary RelianceScope rule)
    ng = [None] * len(F)
    for i, g in enumerate(FG):
        for j in range(i, i + N):
            if g in mw and (ng[j] is None or mw[g] < ng[j]): ng[j] = mw[g]
    last, sz, kin = propagate(seq, 0)
    tk_size = [None] * len(F); tk_first = [False] * len(F)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, F, last, autojunk=False).get_opcodes():
        if op == 'equal':
            for q in range(i2 - i1): tk_size[i1 + q] = sz[j1 + q]; tk_first[i1 + q] = kin[j1 + q] == 0
    return ces, ng, tk_size, tk_first


R = {u: student(u) for u in users}
out = {'n_students': len(users)}
def shares(rule, tau):
    res = []
    for u in users:
        ces, ng, tks, tf = R[u]; size = ng if rule == 'ngram' else tks
        lab = np.array([c if c in 'CES' else ('T' if (s_ is not None and s_ <= tau) else 'X') for c, s_ in zip(ces, size)])
        res.append({c: float((lab == c).mean()) for c in 'CESTX'})
    df = pd.DataFrame(res); o = {}
    for c in 'CESTX':
        v = df[c].values; bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
        o[c] = {'mean': float(v.mean()), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], 'median': float(np.median(v))}
    return o
out['K4_class_shares'] = {f'{rule}_tau{tau}': shares(rule, tau) for rule in ('ngram', 'token') for tau in (5, 20)}
# composition of tokens that entered in large insertions (token rule)
comp = {}
for tau in (5, 20):
    cc = Counter(); xf = Counter()
    for u in users:
        ces, ng, tks, tf = R[u]
        for c, s_, f_ in zip(ces, tks, tf):
            if s_ is None or s_ > tau:
                cc[c] += 1
                if c == 'U': xf['first_snapshot' if f_ else ('not_in_last_snapshot' if s_ is None else 'later')] += 1
    tot = sum(cc.values()); comp[f'tau{tau}'] = {'tokens': tot, **{k: v / tot for k, v in cc.items()},
                                                 'unresolved_by_entry': {k: v / sum(xf.values()) for k, v in xf.items()}}
out['K4_large_insertions_by_evidence'] = comp
out['K4_snapshot_sources'] = {k: int(v) for k, v in S[S.u.isin(users)].src.value_counts().items()}
json.dump(out, open('results/r12_rs.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
