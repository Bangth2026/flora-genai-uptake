"""Analysis log R17, F-28: the unresolved runs of the FLoRA proposals that carry no subtype evidence (X-no-evidence,
src/r16_features.py, thresholds of R16), written to a file for the simulator sensitivity of the benchmark (B-33); F-29, the same partition without the two embedding flags.
The subtypes are computed by src/r16_flora.py, executed here up to its F-20 block; nothing in them changes.
The output contains student text from FLoRA (CC BY 4.0) and is written outside the repository.
Usage: EMB_CACHE=FILE python -I src/r17_flora_xsub.py DATA_DIR R8_TOKEN_CLASSES OUT_CSV [MINLEN] -> OUT_CSV, results/r17_flora_xsub.json"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUTCSV = sys.argv[3]; MINLEN = int(sys.argv[4]) if len(sys.argv) > 4 else 15
src = open(os.path.join(HERE, 'r16_flora.py')).read()
cut = src.index("out['F20_x_subtypes'] = o20")
sys.argv = sys.argv[:3]
G = {'__name__': 'r16_defs', '__file__': os.path.join(HERE, 'r16_flora.py')}
os.chdir(os.path.join(HERE, '..'))
exec(compile(src[:cut], 'r16_flora.py', 'exec'), G)
import pandas as pd
rows = []
for u, sub in G['SUB'].items():
    A = G['EV'][u]['A']; i = 0
    while i < len(sub):
        j = i
        while j < len(sub) and sub[j] == sub[i]: j += 1
        if sub[i] == 'X-no-evidence' and j - i >= MINLEN: rows.append((u, ' '.join(A[i:j])))
        i = j
pd.DataFrame(rows, columns=['user_id', 'text']).to_csv(OUTCSV, index=False)
# F-29: the reporting partition without the two embedding flags (thresholds above 1 switch them off)
import json, numpy as np
F_ = G['F']; EV = G['EV']; SUB2 = {u: F_.x_subtypes(e['cls'], e['aux'], 9.0, 9.0) for u, e in EV.items()}
nX = sum(sum(s is not None for s in v) for v in SUB2.values()); o = {'pooled_share_of_X': {}, 'mean_share_of_proposal': {}, 'students': {}}
for s_ in F_.XSUB:
    o['pooled_share_of_X'][s_] = sum(sum(x == s_ for x in v) for v in SUB2.values()) / nX
    o['mean_share_of_proposal'][s_] = float(np.mean([np.mean([x == s_ for x in SUB2[u]]) for u in EV]))
    o['students'][s_] = int(sum(any(x == s_ for x in SUB2[u]) for u in EV))
INP = G['INP']; o['share_in_logged_pastes'] = {s_: (sum(int(((np.array(SUB2[u]) == s_) & INP[u]).sum()) for u in INP) / max(1, sum(int((np.array(SUB2[u]) == s_).sum()) for u in INP))) for s_ in F_.XSUB}
json.dump({'F29_x_subtypes_without_embedding_flags': o}, open('results/r17_flora_xsub.json', 'w'), indent=1)
print(o['pooled_share_of_X'])
print('runs', len(rows), 'students', len({u for u, _ in rows}))
