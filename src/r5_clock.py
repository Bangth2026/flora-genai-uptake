"""Plan R5, F4: full distribution of ask time minus nearest chat-box Enter keystroke; offsets near 1 h, 10 h, 11 h;
nearest chat-box keystroke of any kind; per-student median offsets. Writes results/r5_clock.json and results/fig_r5_clock.png.
Usage: python -I src/r5_clock.py DATA_DIR"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1]
t = pd.read_pickle(f'{D}/trace.pkl'); d = pd.read_excel(f'{D}/dialogue.xlsx')
ask = t[t.learning_action_label == 'ASK_MULTI_AGENTS'].copy(); ask['t'] = pd.to_datetime(ask.log_trace_time, unit='ms')
ent = ask[ask.event_value.astype(str).str.contains('Enter')]
def nearest(src):
    out, per = [], {}
    for u, x in d.groupby('user_id'):
        ee = np.sort(src[src.user_id == u].t.values)
        if not len(ee): continue
        for a in x.user_ask_time.values:
            dd = (a - ee) / np.timedelta64(1, 's'); dd = dd[np.abs(dd) < 86400]
            if len(dd): v = dd[np.argmin(np.abs(dd))]; out.append(v); per.setdefault(u, []).append(v)
    return np.array(out), per
de, per = nearest(ent); da, _ = nearest(ask)
near = lambda v, c: float((np.abs(np.abs(v) - c) <= 300).mean())
res = {'n_asks': int(len(d)), 'n_matched_enter': int(len(de)), 'share_within_30s': float((np.abs(de) < 30).mean()),
       'quantiles_abs_s': {q: float(np.percentile(np.abs(de), q)) for q in (50, 60, 75, 90, 95)},
       'share_abs_30s_5min': float(((np.abs(de) >= 30) & (np.abs(de) < 300)).mean()), 'share_abs_gt_1h': float((np.abs(de) > 3600).mean()),
       'share_within_5min_of_1h': near(de, 3600), 'share_within_5min_of_10h': near(de, 36000), 'share_within_5min_of_11h': near(de, 39600),
       'any_keystroke': {'n': int(len(da)), 'share_within_30s': float((np.abs(da) < 30).mean()), 'share_within_5min_of_1h': near(da, 3600)},
       'students_with_enter': len(per), 'students_median_abs_gt_30s': int(sum(np.median(np.abs(v)) > 30 for v in per.values())),
       'students_median_near_1h': int(sum(abs(abs(np.median(v)) - 3600) <= 300 for v in per.values())),
       'asks_without_enter_within_30s_but_keystroke_within_30s': None}
# asks whose nearest Enter is > 30 s away: is there any chat-box keystroke within 30 s?
both = []
for u, x in d.groupby('user_id'):
    ee = np.sort(ent[ent.user_id == u].t.values); aa = np.sort(ask[ask.user_id == u].t.values)
    if not len(ee): continue
    for a in x.user_ask_time.values:
        de_ = np.abs((a - ee) / np.timedelta64(1, 's')); da_ = np.abs((a - aa) / np.timedelta64(1, 's'))
        if de_.min() < 86400 and de_.min() >= 30: both.append(da_.min() < 30)
res['asks_without_enter_within_30s_but_keystroke_within_30s'] = float(np.mean(both)) if both else None
res['n_asks_enter_gt_30s'] = len(both)
json.dump(res, open('results/r5_clock.json', 'w'), indent=1); print(json.dumps(res, indent=1))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, ax = plt.subplots(figsize=(6.0, 2.6))
v = np.clip(np.abs(de), 0.01, 86400); bins = np.logspace(-2, np.log10(86400), 60)
ax.hist(v, bins=bins, color='#2a78d6'); ax.set_xscale('log')
for x, lab in [(30, '30 s'), (3600, '1 h'), (36000, '10 h')]: ax.axvline(x, color='#52514e', lw=.7, ls='--'); ax.text(x * 1.08, ax.get_ylim()[1] * .9, lab, fontsize=7, color='#52514e')
ax.set_xlabel('|ask time − nearest chat-box Enter keystroke| (s, log scale)'); ax.set_ylabel('Asks')
for s in ['top', 'right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('results/fig_r5_clock.png', dpi=300)
