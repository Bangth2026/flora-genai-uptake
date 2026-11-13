"""R2 passage labels and third-source sensitivity (ANALYSIS_LOG_R2.md).
Usage: python -I src/r2_label.py DATA_DIR  -> results/r2_passages.pkl, results/r2_sens.json"""
import sys, json, re, warnings
import numpy as np, pandas as pd
from collections import Counter
import statsmodels.api as sm
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2028)
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, n: [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
def mask(P, G, n=8):
    m = np.zeros(len(P), bool)
    for i, g in enumerate(grams(P, n)):
        if g in G: m[i:i + n] = True
    return m
cov = lambda P, G: float(mask(P, G).mean()) if len(P) else 0.0
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').set_index('user_id')
bio = pd.read_excel(f'{D}/bio.xlsx').drop_duplicates('user_id').set_index('user_id')
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
tr = pd.read_pickle(f'{D}/trace.pkl'); an = pd.read_pickle(f'{D}/annot.pkl')
users = sorted(set(d.user_id) & set(p.index)); P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
H = set(g for t in an.highlight_text.dropna() for g in grams(tok(t), 8))
Cc = Counter(g for u in users for g in set(grams(P[u], 8)))
pastes = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy()
pastes['txt'] = pastes.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)

def chatbot_first(u, k):
    x = d[d.user_id == u]; pg = set(grams(P[u], 8)); tbot, tpr = {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s), 8): tbot.setdefault(g, t)
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(s), 8): tpr.setdefault(g, t)
    B = set(tbot); echo = {g for g in B if g in tpr and tpr[g] < tbot[g]}
    cand = {g for g in pg if g in B} - echo - {g for g in pg if g in H or Cc[g] - 1 >= k}
    ted, rem, strs = {}, set(cand), {g: ' ' + ' '.join(g) + ' ' for g in cand}
    for t, c in zip(w[w.user_id == u].t, w[w.user_id == u].writing_content):
        if not rem: break
        s = ' ' + ' '.join(tok(c)) + ' '; hit = [g for g in rem if strs[g] in s]
        for g in hit: ted[g] = t
        rem -= set(hit)
    return {g for g in cand if g in ted and tbot[g] <= ted[g]}, B

# ---- S: third-source threshold sensitivity ----
score = p.reindex(users).proposal_score.values.astype(float); sens = {}; G5 = {}
for k in (3, 5, 10):
    G = {u: chatbot_first(u, k) for u in users}
    if k == 5: G5 = G
    U = np.array([cov(P[u], G[u][0]) for u in users])
    f = sm.OLS(score, sm.add_constant((U - U.mean()) / U.std())).fit(cov_type='HC3')
    sens[k] = {'median': float(np.median(U)), 'mean': float(U.mean()), 'ge25': float((U >= .25).mean()), 'ge50': float((U >= .5).mean()),
               'slope_a': float(f.params[1]), 'ci': [float(v) for v in f.conf_int()[1]], 'p': float(f.pvalues[1])}
    if k == 5: np.save('results/U_R2check.npy', U)
json.dump(sens, open('results/r2_sens.json', 'w'), indent=1); print(json.dumps(sens, indent=1))

# ---- D: passages ----
mt = bio.reindex(users).mother_tongue.astype(str).str.strip().str.lower()
rows = []
for u in users:
    G1, B = G5[u]; PG = set(g for t in pastes[pastes.user_id == u].txt for g in grams(tok(t), 8))
    for j, para in enumerate(str(p.loc[u, 'final_proposal']).split('\n')):
        t = tok(para)
        if len(t) < 25: continue
        rows.append(dict(user_id=u, j=j, text=para.strip(), ntok=len(t), cf=cov(t, G1), paste=cov(t, PG), anybot=cov(t, B),
                         eng=float(mt.loc[u] == 'english') if mt.loc[u] != 'nan' else np.nan))
R = pd.DataFrame(rows)
def lab(r):
    if r.cf >= .8: return 'chatbot'
    if r.cf == 0 and r.paste < .05 and r.anybot == 0: return 'composed'
    return 'mixed'
R['label'] = R.apply(lab, axis=1)
c40 = R[R.ntok >= 40].label.value_counts()
floor = 40 if min(c40.get('chatbot', 0), c40.get('composed', 0)) >= 30 else 25
R = R[R.ntok >= floor].reset_index(drop=True); R.attrs['floor'] = floor
R.to_pickle('results/r2_passages.pkl')
print('floor', floor); print(R.label.value_counts()); print(R.groupby('label').user_id.nunique()); print(R.groupby(['label', 'eng']).size())
