"""Analysis log R12, F-11: share of audited passages (log entry R9 sample, results/r9_audit_sample.csv) that were
selected in the chatbot panel before they entered the editor, by class, with 95% intervals from 2,000 resamples of
students. Usage: python -I src/r12_audit_selection.py -> results/r12_audit_selection.json"""
import json
import numpy as np, pandas as pd
a = pd.read_csv('results/r9_audit_sample.csv'); rng = np.random.default_rng(2150); out = {}
for c, x in a.groupby('cls'):
    us = x.u.unique(); v = x.selected_in_chat_before_entry.astype(float)
    bs = [pd.concat([x[x.u == q] for q in rng.choice(us, len(us))]).selected_in_chat_before_entry.astype(float).mean() for _ in range(2000)]
    out[c] = {'passages': int(len(x)), 'students': int(len(us)), 'selected_in_chat_before_entry': float(v.mean()),
              'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
json.dump(out, open('results/r12_audit_selection.json', 'w'), indent=1); print(out)
