"""Analysis log R25, A-13 (post hoc, written after A-11 and A-12 were seen), from results/r25_coauthor_range.json:
coverage of each range with a tolerance of one and two percentage points; the size of the misses; and coverage when the
text of accepted suggestions shorter than eight tokens (X inserted by a suggestion alone, and T inserted by a suggestion),
which B and R exclude, is added to the upper end. Usage: python -I src/r25_coauthor_range_posthoc.py"""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, '..', 'results')
d = json.load(open(os.path.join(P, 'r25_coauthor_range.json'))); R = pd.DataFrame(d['per_session']); eps = 1e-9
lo, tr = R.C, R.inserted
ups = {'A1-A3, N, B, R': R.C + R.E_after + R.X_with_C, 'without B': R.C + R.E_after + R.X, 'without B and A1': R.C + R.E + R.X,
       'A1-A3, N, B, R plus short-suggestion text': R.C + R.E_after + R.X_with_C + R.X_suggestion_alone + R.T_suggestion}
o = {'coverage_with_tolerance': {k: {str(t): float(((tr >= lo - t - eps) & (tr <= u + t + eps)).mean()) for t in (0, .01, .02)} for k, u in ups.items()}}
b = tr < lo - eps; a = tr > ups['A1-A3, N, B, R'] + eps
o['below_lower_end'] = {'sessions': int(b.sum()), 'mean_points': float(100 * (lo - tr)[b].mean()), 'median_points': float(100 * (lo - tr)[b].median())}
o['above_upper_end_under_B'] = {'sessions': int(a.sum()), 'mean_points': float(100 * (tr - ups['A1-A3, N, B, R'])[a].mean()), 'median_points': float(100 * (tr - ups['A1-A3, N, B, R'])[a].median())}
o['cohort_at_or_above_half'] = {'truth': float((tr >= .5).mean()), **{k: [float((lo >= .5).mean()), float((u >= .5).mean())] for k, u in ups.items()}}
json.dump(o, open(os.path.join(P, 'r25_coauthor_range_posthoc.json'), 'w'), indent=1)
print(json.dumps(o, indent=1))
