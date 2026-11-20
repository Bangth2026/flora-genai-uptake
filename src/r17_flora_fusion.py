"""Analysis log R17, F-30: the evidence-fusion trees with the hyperparameters chosen on the validation set (B-32,
results/r17_fusion_tuned.pkl) applied to the 275 FLoRA proposals without retraining: mean share of chatbot text and
share of students at or above half, for the models trained on 13 and on 26 routes. src/r16_flora.py is executed up to
its F-21 block to rebuild the evidence features; nothing in the attribution changes.
Usage: EMB_CACHE=FILE python -I src/r17_flora_fusion.py DATA_DIR R8_TOKEN_CLASSES -> results/r17_flora_fusion.json"""
import sys, os, json, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(HERE, '..'))
src = open(os.path.join(HERE, 'r16_flora.py')).read(); cut = src.index('# ---------- F-21')
G = {'__name__': 'r16_defs', '__file__': os.path.join(HERE, 'r16_flora.py')}
exec(compile(src[:cut], 'r16_flora.py', 'exec'), G)
EV, boot_mean = G['EV'], G['boot_mean']
T = pickle.load(open('results/r17_fusion_tuned.pkl', 'rb')); out = {'hyperparameters': T['hyperparameters']}
U = np.array([np.mean([c == 'C' for c in EV[u]['cls']]) for u in EV])
for mk in ('gbt_tuned_train_base', 'gbt_tuned_train_all'):
    Uf = np.array([float((T[mk].predict(e['X']) == 0).mean()) for e in EV.values()])
    out[mk] = {'mean_share_C': boot_mean(list(Uf)), 'median_C': float(np.median(Uf)), 'share_students_ge50': boot_mean(list(Uf >= .5))}
    print(mk, out[mk], flush=True)
out['U_mean'] = float(U.mean()); out['U_share_ge50'] = float((U >= .5).mean())
json.dump(out, open('results/r17_flora_fusion.json', 'w'), indent=1, default=float); print('done')
