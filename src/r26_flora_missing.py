"""R26: worst-case bounds for the four FLoRA students absent from the public release.

The data descriptor reports 279 retained students; every file of the public release covers 275.
Nothing is known about the four absent students, so each quantity of the identified range is
recomputed over 279 writers with the four placed at the least favourable value for each end:
for a lower end they count as having no chatbot text, for an upper end as being wholly chatbot
text. Input: results/r23_flora_students.csv (per-writer class shares released with R23).
Output: results/r26_flora_missing.json.
"""
import json
import sys
import pandas as pd

N_REPORTED = 279
src = sys.argv[1] if len(sys.argv) > 1 else 'results/r23_flora_students.csv'
S = pd.read_csv(src)
n = len(S)
miss = N_REPORTED - n

ladder = {
    'L0 lower: C only': S['C'],
    'L1 + E-after': S['C'] + S['E_after'],
    'L2 + E-after + X-with-C': S['C'] + S['E_after'] + S['X_with_C'],
    'L3 + E-after + all X': S['C'] + S['E_after'] + S['X'],
    'L4 + all E + all X': S['C'] + S['E'] + S['X'],
    # L5 counts every token (no assumption), as in R23; class shares need not sum to 1 exactly
    'L5 no assumption: every token': pd.Series(1.0, index=S.index),
}

out = {'n_release': n, 'n_reported': N_REPORTED, 'n_missing': miss, 'ladder': []}
for step, share in ladder.items():
    share = share.clip(upper=1.0)
    row = {'step': step, 'mean_release': float(share.mean()),
           'mean_279_low': float(share.sum() / N_REPORTED),
           'mean_279_high': float((share.sum() + miss) / N_REPORTED)}
    for thr in (25, 50, 75):
        k = int((share >= thr / 100 - 1e-12).sum())
        row[f'at_or_above_{thr}'] = {
            'count_release': k,
            'share_release': k / n,
            'share_279_low': k / N_REPORTED,
            'share_279_high': (k + miss) / N_REPORTED,
        }
    out['ladder'].append(row)

# The identified range reported in the article: [L0, L2] at 50%; wide range [L0, L3]
a = out['ladder']
out['identified_range_50'] = {
    'release': [a[0]['at_or_above_50']['share_release'], a[2]['at_or_above_50']['share_release']],
    'worst_case_279': [a[0]['at_or_above_50']['share_279_low'], a[2]['at_or_above_50']['share_279_high']],
}
out['wide_range_50'] = {
    'release': [a[0]['at_or_above_50']['share_release'], a[3]['at_or_above_50']['share_release']],
    'worst_case_279': [a[0]['at_or_above_50']['share_279_low'], a[3]['at_or_above_50']['share_279_high']],
}
json.dump(out, open('results/r26_flora_missing.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('identified_range_50', 'wide_range_50')}, indent=1))
for r in a:
    print(r['step'], round(r['mean_release'], 4), round(r['mean_279_low'], 4), round(r['mean_279_high'], 4),
          {t: (r[f'at_or_above_{t}']['count_release'], round(r[f'at_or_above_{t}']['share_279_low'], 4),
               round(r[f'at_or_above_{t}']['share_279_high'], 4)) for t in (25, 50, 75)})
