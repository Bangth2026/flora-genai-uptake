"""Plan R5, F8-F9: detector AUCs under alternative paragraph thresholds; size-matched and semantic check of untraced text.
Usage: python -I src/r5_flora_para.py DATA_DIR -> results/r5_flora_para.json
Needs results/r2_scores.pkl (r2_detect.py) and results/_r5_flora_lab.pkl (r5_flora_attr.py)."""
import sys, re, json, pickle, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2053); NB = 2000
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx'); p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
users = sorted(set(d.user_id) & set(p.index)); P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
LAB = pickle.load(open('results/_r5_flora_lab.pkl', 'rb'))
REP = {u: [str(s) for s in d[d.user_id == u].chatbot_utterance_text] for u in users}
R = pd.read_pickle('results/r2_scores.pkl'); R['D3_bino'] = -R['D3_bino']
def pcls(u, text):
    t = tok(text); F = P[u]; n = len(t)
    for i in range(len(F) - n + 1):
        if F[i:i + n] == t: return LAB[u]['R4'][i:i + n]
pc = [pcls(u, s) for u, s in zip(R.user_id, R.text)]
for c in 'CTX': R[c] = [float((a == c).mean()) if a is not None else np.nan for a in pc]
studs = R.user_id.unique(); idx = {u: np.where(R.user_id.values == u)[0] for u in studs}
def cb(fn):
    v = []
    for _ in range(NB):
        s_ = R.iloc[np.concatenate([idx[u] for u in rng.choice(studs, len(studs))])]
        try:
            x = fn(s_)
            if np.isfinite(x): v.append(x)
        except ValueError: pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out = {'n_scored': int(len(R)), 'labels_in_scores': R.label.value_counts().to_dict()}
VARS = {'R4 (chatbot>=.8, T/X>=.9)': (None, .9), 'chatbot>=.9': (.9, .9), 'T/X>=.8': (None, .8), 'T/X>=.95': (None, .95), 'chatbot>=.9, T/X>=.95': (.9, .95)}
for nm, (cth, tth) in VARS.items():
    chat = (R.label == 'chatbot') & ((R.C >= cth) if cth else True)
    cls3 = np.where(chat, 'chatbot', np.where((R.label == 'composed') & (R['T'] >= tth), 'typed', np.where((R.label == 'composed') & (R['X'] >= tth), 'untraced', 'other')))
    R['cls3'] = cls3; o = {'counts': {k: int(v) for k, v in pd.Series(cls3).value_counts().items()}}
    for k in ['D1_ll', 'D2_fdg', 'D3_bino']:
        for a, b in [('chatbot', 'typed'), ('chatbot', 'untraced'), ('untraced', 'typed')]:
            f = lambda s_, a=a, b=b, k=k: roc_auc_score(s_[s_.cls3.isin([a, b])].cls3 == a, s_[s_.cls3.isin([a, b])][k])
            o[f'{k}|{a}_vs_{b}'] = [float(f(R)), cb(f)]
    out[nm] = o
# first language, descriptive (R4 labels)
R['cls3'] = np.where(R.label == 'chatbot', 'chatbot', np.where((R.label == 'composed') & (R['T'] >= .9), 'typed', np.where((R.label == 'composed') & (R['X'] >= .9), 'untraced', 'other')))
sd = {k: R[R.label == 'composed'][k].std() for k in ['D1_ll', 'D2_fdg', 'D3_bino']}
out['l1_descriptive'] = {g: {e: {'paragraphs': int(((R.cls3 == g) & (R.eng == v)).sum()), 'students': int(R[(R.cls3 == g) & (R.eng == v)].user_id.nunique()),
                                  **{k: float(R[(R.cls3 == g) & (R.eng == v)][k].mean() / sd[k]) for k in sd}} for e, v in [('english', 1), ('other', 0)]} for g in ['typed', 'untraced']}
# F9 size-matched containment and semantic similarity
def cont(text, reps, k):
    t = tok(text); G = set(g for s in reps for g in grams(tok(s), k))
    if len(t) < k: return np.nan
    m = np.zeros(len(t), bool)
    for i, g in enumerate(grams(t, k)):
        if g in G: m[i:i + k] = True
    return m.mean()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
SPL = re.compile(r'(?<=[.!?])\s+|\n+')
sents = lambda reps: [s.strip() for r in reps for s in SPL.split(r) if len(tok(s)) >= 5]
EMB = {}
def emb(u):
    if u not in EMB:
        s = sents(REP[u]); EMB[u] = model.encode(s, batch_size=64, normalize_embeddings=True, show_progress_bar=False) if s else np.zeros((0, 384))
    return EMB[u]
res = {}
for cl in ['typed', 'untraced']:
    sub = R[R.cls3 == cl]; v = {'own4': [], 'one4': [], 'own5': [], 'one5': [], 'sem_own': [], 'sem_one': []}
    pe = model.encode([s for s in sub.text], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    for (u, s), e in zip(zip(sub.user_id, sub.text), pe):
        others = rng.choice([x for x in users if x != u], 20, replace=False)
        for k in (4, 5):
            v[f'own{k}'].append(cont(s, REP[u], k)); v[f'one{k}'].append(np.nanmean([cont(s, REP[o], k) for o in others]))
        eo = emb(u); v['sem_own'].append(float((eo @ e).max()) if len(eo) else np.nan)
        v['sem_one'].append(float(np.nanmean([(emb(o) @ e).max() if len(emb(o)) else np.nan for o in others[:5]])))
    res[cl] = {k: float(np.nanmean(x)) for k, x in v.items()} | {'n': int(len(sub))}
    res[cl]['sem_own_minus_one_ci'] = [float(np.percentile([np.nanmean(rng.choice(np.array(v['sem_own']) - np.array(v['sem_one']), len(sub))) for _ in range(NB)], q)) for q in (2.5, 97.5)]
out['paraphrase'] = res
json.dump(out, open('results/r5_flora_para.json', 'w'), indent=1); print(json.dumps(out, indent=1))
