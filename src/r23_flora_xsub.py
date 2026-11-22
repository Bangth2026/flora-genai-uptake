"""Analysis log R23, F-51 (input): per-writer share of the unresolved subtype X-with-C under the reporting partition of
Table 5 (src/r16_flora.py, without the two embedding flags, as in R17 F-29). X-with-C precedes both embedding flags in
the partition, so the window embeddings are not needed; they are replaced by zeros here and nothing else changes.
Usage: python -I src/r23_flora_xsub.py DATA_DIR TOKEN_CLASSES_CSV -> results/r23_flora_xwithc.csv"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r16_features as F
class _Zero:
    def __call__(self, txts): return np.zeros((len(txts), 384), dtype=np.float32)
    def save(self): pass
F.Encoder = _Zero
src = open(os.path.join(HERE, 'r16_flora.py')).read()
cut = src.index('# ---------- F-20 subtypes of X')
G = {'__name__': 'r16_defs', '__file__': os.path.join(HERE, 'r16_flora.py')}
os.chdir(os.path.join(HERE, '..'))
exec(compile(src[:cut], 'r16_flora.py', 'exec'), G)
import pandas as pd
EV = G['EV']; rows = []
for u, e in EV.items():
    sub = F.x_subtypes(e['cls'], e['aux'], 9.0, 9.0)
    rows.append((u, float(np.mean([s == 'X-with-C' for s in sub])), float(np.mean([s is not None for s in sub]))))
df = pd.DataFrame(rows, columns=['user_id', 'X_with_C', 'X'])
df.to_csv('results/r23_flora_xwithc.csv', index=False, float_format='%.6f')
print('agreement with released labels', G['out']['F19_agreement_with_released_labels'], 'mean X-with-C', df.X_with_C.mean(), 'mean X', df.X.mean(), len(df))
