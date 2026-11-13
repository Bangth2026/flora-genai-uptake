"""S1b: provenance of overlapping text from the writing log (ANALYSIS_LOG_S1b.md).
Usage: python -I src/run_s1b_provenance.py DATA_DIR  (needs writing.pkl made from 5.3_writing_log.xlsx)"""
import sys, json, re, numpy as np, pandas as pd
from scipy.stats import spearmanr
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
D = sys.argv[1]
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms')
w = w.sort_values(['user_id', 't'])
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').set_index('user_id')
users = sorted(set(d.user_id) & set(p.index))
OFFS = {'primary_0h': 0, 'plus1h': 1, 'plus4h': 4}
out = {k: {'tok_ue': 0, 'tok_cf': 0, 'Uprov': []} for k in OFFS}
Ue_list, paste_cf, cf_total = [], 0, 0
for u in users:
    x = d[d.user_id == u]
    prop = tok(p.loc[u, 'final_proposal'])
    stu = {tuple(g) for t in x.user_utterance_text for tt in [tok(t)] for g in (tt[i:i+8] for i in range(len(tt) - 7))}
    tbot = {}
    for t, txt in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        tt = tok(txt)
        for i in range(len(tt) - 7):
            g = tuple(tt[i:i+8])
            if g not in stu and g not in tbot: tbot[g] = t
    pos = [(i, tuple(prop[i:i+8])) for i in range(len(prop) - 7)]
    G = {g for _, g in pos if g in tbot}
    # earliest editor time and size of the transition in which the gram first appears
    ted, jump = {}, {}
    ws = w[w.user_id == u]
    remaining, prev_len = set(G), 0
    strs = {g: ' ' + ' '.join(g) + ' ' for g in G}
    for t, c in zip(ws.t, ws.writing_content):
        if not remaining: break
        tt = tok(c); s = ' ' + ' '.join(tt) + ' '
        hit = [g for g in remaining if strs[g] in s]
        for g in hit:
            ted[g] = t; jump[g] = len(tt) - prev_len
        remaining -= set(hit); prev_len = len(tt)
    cov_ue = np.zeros(len(prop), bool)
    for i, g in pos:
        if g in G: cov_ue[i:i+8] = True
    Ue_list.append(cov_ue.mean() if len(prop) else 0.0)
    for k, off in OFFS.items():
        cf = {g for g in G if g not in ted or tbot[g] <= ted[g] + pd.Timedelta(hours=off)}
        cov = np.zeros(len(prop), bool)
        for i, g in pos:
            if g in cf: cov[i:i+8] = True
        out[k]['tok_ue'] += int(cov_ue.sum()); out[k]['tok_cf'] += int(cov.sum())
        out[k]['Uprov'].append(cov.mean() if len(prop) else 0.0)
        if k == 'primary_0h':
            cf_total += len(cf); paste_cf += sum(1 for g in cf if g in jump and jump[g] >= 20)
Ue = np.array(Ue_list)
res = {}
for k in OFFS:
    Up = np.array(out[k]['Uprov'])
    res[k] = {'share_ue_tokens_chatbot_first': out[k]['tok_cf'] / max(out[k]['tok_ue'], 1),
              'spearman_Ue_Uprov': float(spearmanr(Ue, Up)[0]),
              'Uprov_median': float(np.median(Up)), 'Uprov_p75': float(np.percentile(Up, 75)), 'Uprov_p90': float(np.percentile(Up, 90)),
              'share_Uprov_ge_0.25': float((Up >= .25).mean()), 'share_Uprov_ge_0.50': float((Up >= .5).mean())}
    np.save(f'results/Uprov_{k}.npy', Up)
s = res['primary_0h']['share_ue_tokens_chatbot_first']
res['verdict'] = 'Ue valid' if s >= .8 else ('Uprov primary' if s >= .6 else 'Ue not used')
res['paste_share_of_chatbot_first_grams_primary'] = paste_cf / max(cf_total, 1)
res['Ue_check_median'] = float(np.median(Ue))
json.dump(res, open('results/s1b_provenance.json', 'w'), indent=1); print(json.dumps(res, indent=1))
