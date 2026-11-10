"""Post hoc (not in PREREG_R2.md): how much of the 'composed' passages overlaps reading highlights or other students' proposals,
as a rough check on text retyped from shared sources. Usage: python -I src/r2_posthoc.py DATA_DIR -> results/r2_posthoc.json"""
import sys, re, json, numpy as np, pandas as pd
from collections import Counter
D = sys.argv[1]
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, n: [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
def cov(P, G, n=8):
    m = np.zeros(len(P), bool)
    for i, g in enumerate(grams(P, n)):
        if g in G: m[i:i + n] = True
    return m.mean() if len(P) else 0.0
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id'); an = pd.read_pickle(f'{D}/annot.pkl')
R = pd.read_pickle('results/r2_passages.pkl'); C = R[R.label == 'composed']
H = set(g for t in an.highlight_text.dropna() for g in grams(tok(t), 8))
own = {u: set(grams(tok(p.loc[u, 'final_proposal']), 8)) for u in p.index}; cnt = Counter(g for s in own.values() for g in s)
hl = np.array([cov(tok(t), H) for t in C.text]); peer = np.array([cov(tok(t), {g for g in grams(tok(t), 8) if cnt[g] >= 2}) for t in C.text])
S = {'n_composed': int(len(C)), 'share_any_highlight_overlap': float((hl > 0).mean()), 'mean_highlight_cov': float(hl.mean()),
     'share_any_peer_overlap': float((peer > 0).mean()), 'mean_peer_cov': float(peer.mean()), 'share_peer_cov_ge20': float((peer >= .2).mean())}
json.dump(S, open('results/r2_posthoc.json', 'w'), indent=1); print(S)
