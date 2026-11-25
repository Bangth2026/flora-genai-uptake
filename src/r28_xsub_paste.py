"""Analysis log R28, F-66 (FLoRA): share of each subtype of the unresolved outcome lying inside logged pasted text, with
the logged pastes of the whole trace (DATA_FULL/pastes.pkl, all eight sheets) instead of those of the first sheet of each
workbook (68 students; Table 5 of earlier versions). Subtypes as in src/r23_flora_xsub.py (the partition of Table 5,
without the embedding flags); a token lies inside logged pasted text if one of its covering 8-grams occurs in the text of
one of the writer's logged pastes (src/r10_flora_checks.py, P-1).
Usage: python -I src/r28_xsub_paste.py DATA_FULL_DIR -> results/r28_xsub_paste.json"""
import sys, os, re, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
D = sys.argv[1]
import r16_features as F
class _Zero:
    def __call__(self, txts): return np.zeros((len(txts), 384), dtype=np.float32)
    def save(self): pass
F.Encoder = _Zero
src = open(os.path.join(HERE, 'r16_flora.py')).read()
cut = src.index('# ---------- F-20 subtypes of X')
G = {'__name__': 'r16_defs', '__file__': os.path.join(HERE, 'r16_flora.py')}
os.chdir(os.path.join(HERE, '..')); sys.argv = [os.path.join(HERE, 'r16_flora.py'), D, 'results/flora_token_classes.csv.gz']
exec(compile(src[:cut], 'r16_flora.py', 'exec'), G)
EV = G['EV']
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower()); grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
pa = pd.read_pickle(f'{D}/pastes.pkl'); pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
PG = {u: {g for t in x.txt for g in grams(tok(t), 8)} for u, x in pa.groupby('user_id')}
tot, inp = {}, {}
for u, e in EV.items():
    if u not in PG: continue
    A = [str(a) for a in e['A']]; m = np.zeros(len(A), bool)
    for i, g in enumerate(grams(A, 8)):
        if g in PG[u]: m[i:i + 8] = True
    sub = F.x_subtypes(e['cls'], e['aux'], 9.0, 9.0)
    for s_, c_, mm in zip(sub, e['cls'], m):
        k = s_ if s_ is not None else c_
        tot[k] = tot.get(k, 0) + 1; inp[k] = inp.get(k, 0) + int(mm)
    xs = [s_ is not None for s_ in sub]
    tot['X'] = tot.get('X', 0) + int(np.sum(xs)); inp['X'] = inp.get('X', 0) + int(np.sum(np.array(xs) & m))
out = {'students_with_logged_pastes': len(set(PG) & set(EV)), 'share_in_logged_pastes': {k: inp[k] / tot[k] for k in tot}, 'tokens': tot}
json.dump(out, open('results/r28_xsub_paste.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
