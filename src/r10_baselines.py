"""Analysis log R10, W-1: winnowing and greedy string tiling of the FLoRA proposals against the student's own chatbot
replies and against other students' replies (chance level); share of marked tokens by five-class label.
Usage: python -I src/r10_baselines.py DATA_DIR -> results/r10_baselines_flora.json"""
import sys, os, json
from collections import Counter
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demo'))
from provenance import tokens
from baselines import winnow_mark, tile_mark, overlap_mark
D = sys.argv[1]; rng = np.random.default_rng(2102)
d = pd.read_excel(f'{D}/dialogue.xlsx')
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])
LAB = {u: np.array(g['class'].tolist()) for u, g in tc.groupby('user_id')}; FT = {u: g.token.astype(str).tolist() for u, g in tc.groupby('user_id')}
REP = {u: [tokens(s) for s in g.chatbot_utterance_text.astype(str)] for u, g in d.groupby('user_id')}
users = sorted(set(LAB) & set(REP)); U = np.array([(LAB[u] == 'C').mean() for u in users])
out = {'n_students': len(users), 'Ustar_median': float(np.median(U)), 'Ustar_ge50': float((U >= .5).mean())}
for name, fn in (('naive_overlap_8gram', lambda F, s: overlap_mark(F, s, 8)), ('winnowing_k8_w4', lambda F, s: winnow_mark(F, s, 8, 4)), ('string_tiling_mml8', lambda F, s: tile_mark(F, s, 8))):
    own, chance, by = [], [], Counter()
    for u in users:
        F = FT[u]; m = np.array(fn(F, REP[u]), bool); own.append(m.mean())
        for c in 'CESTX': by[c] += int((m & (LAB[u] == c)).sum())
        others = rng.choice([v for v in users if v != u], 20, replace=False); chance.append(np.median([np.mean(fn(F, REP[v])) for v in others]))
    own, chance = np.array(own), np.array(chance); tot = sum(by.values())
    out[name] = {'median_share': float(np.median(own)), 'mean_share': float(own.mean()), 'ge50': float((own >= .5).mean()), 'ge25': float((own >= .25).mean()),
                 'chance_median_of_medians': float(np.median(chance)), 'chance_mean': float(chance.mean()),
                 'spearman_with_Ustar': float(stats.spearmanr(own, U)[0]), 'marked_tokens_by_class': {c: by[c] / tot for c in 'CESTX'},
                 'share_of_marked_not_C': 1 - by['C'] / tot}
    print(name, out[name], flush=True)
json.dump(out, open('results/r10_baselines_flora.json', 'w'), indent=1)
