"""Clock check from chat-box Enter keystrokes (trace log, platform clock) vs dialogue ask times.
Usage: python -I src/r1_clock.py DATA_DIR  (needs trace.pkl from Figshare 5.2 files)"""
import sys, json, numpy as np, pandas as pd
D = sys.argv[1]
t = pd.read_pickle(f'{D}/trace.pkl'); d = pd.read_excel(f'{D}/dialogue.xlsx')
e = t[(t.learning_action_label == 'ASK_MULTI_AGENTS') & t.event_value.astype(str).str.contains('Enter')].copy()
e['t'] = pd.to_datetime(e.log_trace_time, unit='ms')
diffs = []
for u, x in d.groupby('user_id'):
    ee = np.sort(e[e.user_id == u].t.values)
    if not len(ee): continue
    for a in x.user_ask_time.values:
        dd = (a - ee) / np.timedelta64(1, 's'); dd = dd[np.abs(dd) < 86400]
        if len(dd): diffs.append(dd[np.argmin(np.abs(dd))])
diffs = np.array(diffs)
res = {'n_matched': int(len(diffs)), 'n_within_30s': int((np.abs(diffs) < 30).sum()), 'median_s': float(np.median(diffs)),
       'p25_s': float(np.percentile(diffs, 25)), 'p75_s': float(np.percentile(diffs, 75))}
json.dump(res, open('results/r1_clock.json', 'w'), indent=1); print(res)
