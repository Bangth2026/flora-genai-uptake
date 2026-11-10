"""Clock alignment between writing log (epoch ms) and dialogue timestamps.
Text a student pastes into a prompt from the editor must already be in the editor at ask time.
For each candidate offset, report the share of such pasted 8-grams whose first editor appearance
precedes the ask time. Uses no outcome or uptake variable."""
import sys, re, numpy as np, pandas as pd
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
w = pd.read_pickle(sys.argv[1] + '/writing.pkl'); d = pd.read_excel(sys.argv[1] + '/dialogue.xlsx')
w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
rows = []
for u, x in d.groupby('user_id'):
    ws = w[w.user_id == u]
    if ws.empty: continue
    snaps = [(t, ' ' + ' '.join(tok(c)) + ' ') for t, c in zip(ws.t, ws.writing_content)]
    for _, r in x.iterrows():
        tt = tok(r.user_utterance_text)
        if len(tt) < 40: continue  # only long prompts, likely pasted text
        grams = [' ' + ' '.join(tt[i:i+8]) + ' ' for i in range(0, len(tt) - 7, 8)]
        for g in grams[:10]:
            first = next((t for t, s in snaps if g in s), None)
            if first is not None: rows.append((r.user_ask_time, first))
a = pd.DataFrame(rows, columns=['ask', 'ed'])
print('pasted grams found in editor:', len(a))
for off in range(-12, 13, 1):
    print(off, round(((a.ask + pd.Timedelta(hours=off)) >= a.ed).mean(), 3))
