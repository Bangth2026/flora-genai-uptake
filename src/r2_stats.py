"""R2 detector analyses D-a..D-e (ANALYSIS_LOG_R2.md). Usage: python -I src/r2_stats.py -> results/r2_stats.json"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(2029); B = 2000
R = pd.read_pickle('results/r2_scores.pkl'); Pp = pd.read_pickle('results/r2_proposal_scores.pkl')
R['D3_bino'] = -R['D3_bino']; Pp['D3_bino'] = -Pp['D3_bino']   # higher = more machine-like for all three
DET = ['D1_ll', 'D2_fdg', 'D3_bino']; R['y'] = (R.label == 'chatbot').astype(int)
U = np.load('results/U_R1.npy'); users = sorted(Pp.user_id); Pp = Pp.set_index('user_id').loc[users]
studs = R.user_id.unique(); idx = {u: np.where(R.user_id.values == u)[0] for u in studs}
def cboot(fn):
    v = []
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(studs, len(studs))])
        try:
            x = fn(R.iloc[ii])
            if np.isfinite(x): v.append(x)
        except ValueError: pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
ci_ = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
S = {'n': {'chatbot': int(R.y.sum()), 'composed': int((1 - R.y).sum()), 'students_chatbot': int(R[R.y == 1].user_id.nunique()),
           'students_composed': int(R[R.y == 0].user_id.nunique()), 'composed_eng': int(((R.y == 0) & (R.eng == 1)).sum()),
           'composed_other': int(((R.y == 0) & (R.eng == 0)).sum()), 'composed_students_eng': int(R[(R.y == 0) & (R.eng == 1)].user_id.nunique())},
     'ntok_median': {k: float(R[R.label == k].ntok.median()) for k in ['chatbot', 'composed']}}
for k in DET:
    s = {}
    s['auc'] = float(roc_auc_score(R.y, R[k])); s['auc_ci'] = cboot(lambda r: roc_auc_score(r.y, r[k]))
    s['auc_len_only'] = float(roc_auc_score(R.y, np.log(R.ntok)))
    for tpr in (50, 80):
        th = np.percentile(R[R.y == 1][k], 100 - tpr)
        ff = lambda r, e=None: ((r.y == 0) & (r[k] >= th) & (True if e is None else r.eng == e)).sum() / ((r.y == 0) & (True if e is None else r.eng == e)).sum()
        s[f'tpr{tpr}'] = {'threshold': float(th), 'false_flag': float(ff(R)), 'ff_eng': float(ff(R, 1)), 'ff_other': float(ff(R, 0)),
                          'diff_other_minus_eng_ci': cboot(lambda r: ff(r, 0) - ff(r, 1))}
    both = R.groupby('user_id').y.nunique(); both = both[both == 2].index
    pdiff = np.array([R[(R.user_id == u) & (R.y == 1)][k].mean() - R[(R.user_id == u) & (R.y == 0)][k].mean() for u in both])
    s['within'] = {'n_students': int(len(both)), 'mean_diff': float(pdiff.mean()), 'share_positive': float((pdiff > 0).mean()),
                   'ci': ci_([rng.choice(pdiff, len(pdiff)).mean() for _ in range(B)]),
                   'sd_composed': float(R[R.y == 0][k].std())}
    comp = R[R.y == 0]
    s['composed_by_l1'] = {'mean_eng': float(comp[comp.eng == 1][k].mean()), 'mean_other': float(comp[comp.eng == 0][k].mean()),
                           'diff_ci': cboot(lambda r: r[(r.y == 0) & (r.eng == 0)][k].mean() - r[(r.y == 0) & (r.eng == 1)][k].mean()),
                           'logntok_eng': float(np.log(comp[comp.eng == 1].ntok).mean()), 'logntok_other': float(np.log(comp[comp.eng == 0].ntok).mean())}
    hi, lo = U >= .5, U < .05; m = hi | lo
    r_ = spearmanr(Pp[k].values, U)
    s['proposal'] = {'spearman': float(r_[0]), 'p': float(r_[1]), 'auc_ge50_vs_lt05': float(roc_auc_score(hi[m], Pp[k].values[m])),
                     'n_ge50': int(hi.sum()), 'n_lt05': int(lo.sum())}
    S[k] = s
json.dump(S, open('results/r2_stats.json', 'w'), indent=1); print(json.dumps(S, indent=1))
