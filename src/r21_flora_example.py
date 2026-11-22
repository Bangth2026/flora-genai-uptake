"""Analysis log R21, F-48: what labelled FLoRA proposals look like (Fig. 5 of the article). Four real proposals are
chosen by a rule fixed before any was drawn: the proposals whose chatbot-first share U is nearest to the 25th, 50th,
75th and 90th percentiles of U across the 275 students (ties: length nearest the median length). Each is drawn token by
token in reading order, 60 tokens per row, coloured by its released label (results/flora_token_classes.csv.gz). No text
is shown and no student is identified. Usage: python -I src/r21_flora_example.py OUT.png -> OUT.png,
results/r21_flora_example.json"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; WROW = 60
tc = pd.read_csv(os.path.join(HERE, '..', 'results', 'flora_token_classes.csv.gz')).sort_values(['user_id', 'token_index'])
sc = pd.read_csv(os.path.join(HERE, '..', 'results', 'flora_student_classes.csv')).set_index('user_id')
L = tc.groupby('user_id').size().reindex(sc.index); medL = L.median()
picks = []
for q in (.25, .50, .75, .90):
    target = sc.C.quantile(q); dist = (sc.C - target).abs()
    cand = dist[dist <= dist.min() + 1e-12].index
    u = min(cand, key=lambda v: (abs(L[v] - medL), v)); picks.append((q, int(u), float(target)))
col = {'C': '#1baf7a', 'E': '#eb6834', 'S': '#8a8984', 'T': '#2a78d6', 'X': '#b9a6e6'}
name = {'C': 'C chatbot-first', 'E': 'E echo', 'S': 'S shared source', 'T': 'T typed', 'X': 'X unresolved'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})
rows = [int(np.ceil(L[u] / WROW)) for _, u, _ in picks]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 0.16 * max(rows) + 1.1), gridspec_kw={'wspace': 0.08})
res = {'rule': 'U nearest the 25th, 50th, 75th, 90th percentile; ties by length nearest the median', 'proposals': []}
for ax, (q, u, target), nr in zip(axes, picks, rows):
    c = tc[tc.user_id == u]['class'].tolist()
    for i, k in enumerate(c):
        r, x = divmod(i, WROW); ax.add_patch(Rectangle((x, -r), 1, 0.86, fc=col[k], ec='none'))
    ax.set_xlim(0, WROW); ax.set_ylim(-max(rows), 1); ax.axis('off')
    s = sc.loc[u]
    ax.set_title(f'{int(q * 100)}th percentile of U\n{len(c)} tokens: C {100*s.C:.0f}%, X {100*s.X:.0f}%', fontsize=7, loc='left')
    res['proposals'].append({'percentile': q, 'U_target': target, 'tokens': len(c), **{k: float(s[k]) for k in ('C', 'E', 'S', 'T', 'X', 'X_with_C')}})
fig.legend(handles=[Patch(fc=col[k], label=name[k]) for k in 'CESTX'], loc='lower center', ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.savefig(OUT, dpi=300, bbox_inches='tight')
json.dump(res, open(os.path.join(HERE, '..', 'results', 'r21_flora_example.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
