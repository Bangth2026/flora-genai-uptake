"""Post hoc refinement of R4 F-1 (not in PREREG_R4): a logged paste explains an n-gram's entry only if the paste
occurred at its first editor appearance. Logged pastes also contain text the student had typed earlier and then
moved within the document. Usage: python -I src/r4_flora_tx_timing.py DATA_DIR -> results/r4_flora_tx_timing.json"""
import sys, re, json
import numpy as np, pandas as pd
from difflib import SequenceMatcher
D = sys.argv[1]; NET = 5; WIN = pd.Timedelta(seconds=10)
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k=8: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
tr = pd.read_pickle(f'{D}/trace.pkl'); pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy()
pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False); pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms')
res = {'T': [0, 0, 0], 'X': [0, 0, 0]}  # [grams, in any paste, pasted at entry]
for u in sorted(set(pa.user_id) & set(p.index) & set(w.user_id)):
    F = tok(p.loc[u, 'final_proposal']); need = set(grams(F)); ted, mode, prev = {}, {}, []
    for t, c in zip(w[w.user_id == u].t, w[w.user_id == u].writing_content):
        tk = tok(c)
        if need:
            new = [g for g in set(grams(tk)) if g in need]
            if new:
                ins = sum(j2 - j1 for op, i1, i2, j1, j2 in SequenceMatcher(None, prev, tk, autojunk=False).get_opcodes() if op in ('insert', 'replace'))
                for g in new: ted[g] = t; mode[g] = 'T' if ins <= NET else 'X'; need.discard(g)
        prev = tk
    pt = {}
    for t, s in zip(pa[pa.user_id == u].t, pa[pa.user_id == u].txt):
        for g in grams(tok(s)): pt.setdefault(g, []).append(t)
    for g, m in mode.items():
        r = res[m]; r[0] += 1
        if g in pt:
            r[1] += 1
            if any(abs(t - ted[g]) <= WIN for t in pt[g]): r[2] += 1
out = {c: {'grams': v[0], 'in_any_logged_paste': v[1] / v[0], 'pasted_at_entry': v[2] / v[0]} for c, v in res.items()}
out['clock_check_example'] = str(pa.t.iloc[0]); json.dump(out, open('results/r4_flora_tx_timing.json', 'w'), indent=1); print(out)
