"""Analysis log R10, P-1 to P-3 (FLoRA): paste checks with student-level intervals, completeness of the editor log for
students with and without logged pastes, and asks whose nearest Enter keystroke lies near 1, 10 or 11 h.
Token labels: results/flora_token_classes.csv.gz (primary attribution).
Usage: python -I src/r10_flora_checks.py DATA_DIR -> results/r10_flora_checks.json"""
import sys, json, re
import numpy as np, pandas as pd
from scipy import stats
D = sys.argv[1]; rng = np.random.default_rng(2103); NB = 2000
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower()); grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])
LAB = {u: np.array(g['class'].tolist()) for u, g in tc.groupby('user_id')}; FT = {u: g.token.astype(str).tolist() for u, g in tc.groupby('user_id')}
users = sorted(LAB)
tr = pd.read_pickle(f'{D}/trace.pkl')
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms')
ask = tr[tr.learning_action_label == 'ASK_MULTI_AGENTS'].copy(); ask['t'] = pd.to_datetime(ask.log_trace_time, unit='ms'); del tr
def cover(F, G, k=8):
    m = np.zeros(len(F), bool)
    for i, g in enumerate(grams(F, k)):
        if g in G: m[i:i + k] = True
    return m
PU = sorted(set(pa.user_id) & set(users)); rows = []
for u in PU:
    m = cover(FT[u], {g for t in pa[pa.user_id == u].txt for g in grams(tok(t), 8)}); L = LAB[u]
    rows.append({**{f'{c}_n': int((L == c).sum()) for c in 'CTX'}, **{f'{c}_in': int(((L == c) & m).sum()) for c in 'CTX'}, 'u': u})
r = pd.DataFrame(rows)
def ratio_ci(a, b):
    v = []
    for _ in range(NB):
        i = rng.integers(0, len(r), len(r)); v.append(r[a].values[i].sum() / max(r[b].values[i].sum(), 1))
    return {'pooled': float(r[a].sum() / r[b].sum()), 'ci': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            'students_with_class': int((r[b] > 0).sum()), 'student_median': float((r[a] / r[b]).dropna().median())}
out = {'n_students_with_pastes': len(PU), 'P1_token_coverage_by_logged_pastes': {c: ratio_ci(f'{c}_in', f'{c}_n') for c in 'CTX'}}
# entry-level version from R9 F-5 (per-student counts are recomputed there; here intervals for the published rates)
try:
    r9 = json.load(open('results/r9_flora.json'))['F5_tau_window']['5']['win10']; out['P1_entry_level_rates_R9'] = r9
except Exception: pass
# P-2 completeness of the editor log
w = pd.read_pickle(f'{D}/writing.pkl'); w = w[w.user_id.isin(users)].sort_values(['user_id', 'writing_log_time'])
q = []
for u, g in w.groupby('user_id'):
    t = g.writing_log_time.values.astype(float) / 1e3; dt = np.diff(t); dts = dt[dt < 1800]
    prev = []; large = 0; changes = 0
    for c in g.writing_content.astype(str):
        cur = tok(c); m_ = min(len(prev), len(cur)); pre = 0
        while pre < m_ and prev[pre] == cur[pre]: pre += 1
        suf = 0
        while suf < m_ - pre and prev[-1 - suf] == cur[-1 - suf]: suf += 1
        ins = len(cur) - pre - suf
        if ins > 0: changes += 1; large += ins > 5
        prev = cur
    q.append(dict(u=u, paste=u in set(PU), snaps=len(g), per100=len(g) / max(len(FT[u]), 1) * 100, med_dt=float(np.median(dts)) if len(dts) else np.nan,
                  sessions=int((dt >= 1800).sum()) + 1, large_share=large / max(changes, 1), writing_hours=float(dts.sum() / 3600)))
q = pd.DataFrame(q); P2 = {}
for col in ('snaps', 'per100', 'med_dt', 'sessions', 'large_share', 'writing_hours'):
    a, b = q[q.paste][col].dropna(), q[~q.paste][col].dropna()
    P2[col] = {'with_pastes_median': float(a.median()), 'without_pastes_median': float(b.median()), 'mannwhitney_p': float(stats.mannwhitneyu(a, b).pvalue)}
out['P2_editor_log_completeness'] = P2 | {'n_with': int(q.paste.sum()), 'n_without': int((~q.paste).sum())}
# P-3 asks near 1, 10, 11 h from their nearest Enter keystroke
d = pd.read_excel(f'{D}/dialogue.xlsx'); ent = ask[ask.event_value.astype(str).str.contains('Enter')]; P3 = {}
for h in (1, 10, 11):
    n_, st, kw = 0, set(), 0
    for u, x in d.groupby('user_id'):
        ee = np.sort(ent[ent.user_id == u].t.values); aa = np.sort(ask[ask.user_id == u].t.values)
        if not len(ee): continue
        for a in x.user_ask_time.values:
            dd = (a - ee) / np.timedelta64(1, 's'); dd = dd[np.abs(dd) < 86400]
            if len(dd) and abs(abs(dd[np.argmin(np.abs(dd))]) - 3600 * h) <= 300:
                n_ += 1; st.add(u); kw += bool(len(aa)) and np.min(np.abs((a - aa) / np.timedelta64(1, 's'))) < 30
    P3[f'{h}h'] = {'asks': n_, 'students': len(st), 'asks_with_any_chatbox_keystroke_within_30s': kw}
out['P3_clock_offsets'] = P3
print(json.dumps(out, indent=1, default=float)); json.dump(out, open('results/r10_flora_checks.json', 'w'), indent=1, default=float)
