"""Analysis log R10, addendum A-1: 8-grams that meet two ordered rules at once, and tokens covered by both C and E.
Usage: python -I src/r10_rule_conflicts.py DATA_DIR RS_DIR -> results/r10_rule_conflicts.json"""
import sys, re, json
from collections import Counter
import numpy as np, pandas as pd
D, RSD = sys.argv[1], sys.argv[2]
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower()); grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
out = {}
# FLoRA
d = pd.read_excel(f'{D}/dialogue.xlsx'); an = pd.read_pickle(f'{D}/annot.pkl')
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])
FT = {u: g.token.astype(str).tolist() for u, g in tc.groupby('user_id')}; LAB = {u: g['class'].tolist() for u, g in tc.groupby('user_id')}
HL = {g for s in an.highlight_text.dropna() for g in grams(tok(s), 8)}
CC = Counter(g for u in FT for g in set(grams(FT[u], 8)))
pf_sh = pf = n_all = 0
for u, x in d.groupby('user_id'):
    if u not in FT: continue
    tb, tp = {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s), 8): tb.setdefault(g, t)
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(s), 8): tp.setdefault(g, t)
    for g in set(grams(FT[u], 8)):
        n_all += 1
        if g in tb and g in tp and tp[g] < tb[g]:
            pf += 1; pf_sh += (g in HL or CC[g] - 1 >= 5)
out['flora'] = {'final_8grams': n_all, 'prompt_first_8grams': pf, 'prompt_first_and_shared': pf_sh}
# tokens covered by both C and E n-grams: from the primary labels, a token is C-and-E-covered if its class is C and an
# E-before-C projection (results/r5 analysis) changes it; reported there as 3.7% of tokens. Recomputed here from labels:
# runs of C adjacent to E within n - 1 tokens bound the conflict region.
conf = tot = 0
for u in FT:
    L = LAB[u]
    for i, c in enumerate(L):
        tot += 1
        if c == 'C' and any(L[j] == 'E' for j in range(max(0, i - 7), min(len(L), i + 8))): conf += 1
out['flora']['C_tokens_within_7_of_E'] = conf / tot
# RelianceScope: prompt-first and template
Lj = lambda f: pd.DataFrame(list(json.load(open(f'{RSD}/logs/{f}')).values())[0])
msg, ed, conv = Lj('messages.json'), Lj('edit logs.json'), Lj('conversations.json')
c2u = dict(zip(conv.id, conv.userId)); msg['u'] = msg.conversationId.map(c2u); ed['u'] = ed.conversationId.map(c2u)
template = Counter(ed.sort_values('createdAt').groupby('u').code.first()).most_common(1)[0][0]; TG = set(grams(tok(template), 8))
pf_sh = pf = n_all = 0
for u, x in msg.dropna(subset=['u']).groupby('u'):
    e = ed[ed.u == u]
    if not len(e): continue
    F = tok(e.sort_values('createdAt').code.iloc[-1]); tb, tp = {}, {}
    for t, r, c in zip(x.createdAt, x.role, x.content):
        for g in grams(tok(c), 8): (tb if r == 'assistant' else tp).setdefault(g, t)
    for g in set(grams(F, 8)):
        n_all += 1
        if g in tb and g in tp and tp[g] < tb[g]: pf += 1; pf_sh += g in TG
out['reliancescope'] = {'final_8grams': n_all, 'prompt_first_8grams': pf, 'prompt_first_and_template': pf_sh}
r8 = json.load(open('results/r8_rs.json')); out['editor_first_and_shared_R8'] = r8.get('K2_E_before_S', {}).get('S_grams_editor_before_reply')
print(json.dumps(out, indent=1)); json.dump(out, open('results/r10_rule_conflicts.json', 'w'), indent=1)
