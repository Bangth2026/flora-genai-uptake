"""Plan R4, part F: FLoRA edit-size entry rule, T/X validation against logged pastes, class shares with intervals,
paragraph classes and detectors under the new rule, and the post hoc paraphrase check of untraced text.
Usage: python -I src/r4_flora.py DATA_DIR -> results/r4_flora_stats.json"""
import sys, re, json, warnings
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2041); NB = 2000; NET = 5
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k=8: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
def cover(F, G, k=8):
    m = np.zeros(len(F), bool)
    for i, g in enumerate(grams(F, k)):
        if g in G: m[i:i + k] = True
    return m
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
an = pd.read_pickle(f'{D}/annot.pkl'); tr = pd.read_pickle(f'{D}/trace.pkl')
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
users = sorted(set(d.user_id) & set(p.index)); P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
H = set(g for t in an.highlight_text.dropna() for g in grams(tok(t))); Cc = Counter(g for u in users for g in set(grams(P[u])))
RANK = {'C': 0, 'E': 1, 'S': 2, 'T': 3, 'X': 4}; INV = {v: k for k, v in RANK.items()}
REP = {u: [str(s) for s in d[d.user_id == u].chatbot_utterance_text] for u in users}

def classes(u):
    x = d[d.user_id == u]; FG = grams(P[u]); need = set(FG); tbot, tpr = {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s)): tbot.setdefault(g, pd.Timestamp(t))
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(s)): tpr.setdefault(g, pd.Timestamp(t))
    ted, mnet, medit, prev = {}, {}, {}, []
    for t, c in zip(w[w.user_id == u].t, w[w.user_id == u].writing_content):
        tk = tok(c)
        if need:
            new = [g for g in set(grams(tk)) if g in need]
            if new:
                net = abs(len(tk) - len(prev))
                ins = sum(j2 - j1 for op, i1, i2, j1, j2 in SequenceMatcher(None, prev, tk, autojunk=False).get_opcodes() if op in ('insert', 'replace'))
                for g in new: ted[g] = t; mnet[g] = 'T' if net <= NET else 'X'; medit[g] = 'T' if ins <= NET else 'X'; need.discard(g)
        prev = tk
    out = {}
    for nm, mode in [('edit', medit), ('net', mnet)]:
        cls = {}
        for g in set(FG):
            bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]; shared = g in H or Cc[g] - 1 >= 5
            if bot and not echo and not shared and g in ted and tbot[g] <= ted[g]: cls[g] = 'C'
            elif bot and (echo or (g in ted and ted[g] < tbot[g])): cls[g] = 'E'
            elif shared: cls[g] = 'S'
            else: cls[g] = mode.get(g, 'X')
        best = np.full(len(P[u]), 9)
        for i, g in enumerate(FG): best[i:i + 8] = np.minimum(best[i:i + 8], RANK[cls[g]])
        out[nm] = np.array([INV.get(b, 'X') for b in best])
    return out

LAB = {}
for k, u in enumerate(users):
    LAB[u] = classes(u)
    if k % 50 == 0: print(k, flush=True)
U1 = np.load('results/U_R1.npy')
bci = lambda f, v: [float(np.percentile([f(rng.choice(v, len(v))) for _ in range(NB)], q)) for q in (2.5, 97.5)]
out = {'C_matches_Ustar': float(np.abs(np.array([(LAB[u]['edit'] == 'C').mean() for u in users]) - U1).max())}
for nm in ['edit', 'net']:
    sh = pd.DataFrame({c: [(LAB[u][nm] == c).mean() for u in users] for c in 'CESTX'})
    out[f'classes_{nm}'] = {c: {'mean': float(sh[c].mean()), 'median': float(sh[c].median()), 'mean_ci': bci(np.mean, sh[c].values)} for c in 'CESTX'}
# F-1 validation against logged pastes
tu = sorted(set(pa.user_id) & set(users)); PG = {u: set(g for t in pa[pa.user_id == u].txt for g in grams(tok(t))) for u in tu}
for nm in ['edit', 'net']:
    cnt = {c: [0, 0] for c in 'TX'}
    for u in tu:
        m = cover(P[u], PG[u]); lab = LAB[u][nm]
        for c in 'TX': cnt[c][0] += int(((lab == c) & m).sum()); cnt[c][1] += int((lab == c).sum())
    out[f'validation_{nm}'] = {c: {'in_logged_paste': cnt[c][0] / max(cnt[c][1], 1), 'tokens': cnt[c][1]} for c in 'TX'} | {'n_students': len(tu)}
# F-3 paragraph classes and detectors under the edit rule
R = pd.read_pickle('results/r2_scores.pkl')
def pcls(u, text):
    t = tok(text); F = P[u]; n = len(t)
    for i in range(len(F) - n + 1):
        if F[i:i + n] == t: return LAB[u]['edit'][i:i + n]
pc = [pcls(u, s) for u, s in zip(R.user_id, R.text)]
for c in 'TX': R[c] = [float((a == c).mean()) if a is not None else np.nan for a in pc]
R['cls3'] = np.where(R.label == 'chatbot', 'chatbot', np.where((R.label == 'composed') & (R['T'] >= .9), 'typed', np.where((R.label == 'composed') & (R['X'] >= .9), 'untraced', 'other')))
comp = R[R.label == 'composed']
out['paragraphs'] = {'counts': R.cls3.value_counts().to_dict(), 'composed_typed_share': float((comp['T'] >= .9).mean()), 'composed_untraced_share': float((comp['X'] >= .9).mean()),
                     'typed_by_l1': {k: float((comp[comp.eng == v]['T'] >= .9).mean()) for k, v in [('eng', 1), ('other', 0)]},
                     'typed_eng_paras': int(((R.cls3 == 'typed') & (R.eng == 1)).sum()), 'typed_eng_students': int(R[(R.cls3 == 'typed') & (R.eng == 1)].user_id.nunique())}
R['D3_bino'] = -R['D3_bino']; studs = R.user_id.unique(); idx = {u: np.where(R.user_id.values == u)[0] for u in studs}
def cb(fn):
    v = []
    for _ in range(NB):
        s_ = R.iloc[np.concatenate([idx[u] for u in rng.choice(studs, len(studs))])]
        try:
            x = fn(s_)
            if np.isfinite(x): v.append(x)
        except (ValueError, ZeroDivisionError): pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def auc(s_, a, b, k):
    m = s_.cls3.isin([a, b]); return roc_auc_score(s_[m].cls3 == a, s_[m][k])
for k in ['D1_ll', 'D2_fdg', 'D3_bino']:
    o = {}
    for a, b in [('chatbot', 'typed'), ('chatbot', 'untraced'), ('untraced', 'typed')]: o[f'{a}_vs_{b}'] = [float(auc(R, a, b, k)), cb(lambda s_: auc(s_, a, b, k))]
    sd = R[R.label == 'composed'][k].std(); f = lambda s_: (s_[(s_.cls3 == 'typed') & (s_.eng == 0)][k].mean() - s_[(s_.cls3 == 'typed') & (s_.eng == 1)][k].mean()) / sd
    o['l1_diff_sd_typed'] = [float(f(R)), cb(f)]
    m = R[R.cls3.isin(['typed', 'untraced'])].dropna(subset=['eng']).copy(); m['z'] = (m[k] - m[k].mean()) / m[k].std(); m['untr'] = (m.cls3 == 'untraced').astype(int); m['lnt'] = np.log(m.ntok)
    fit = smf.ols('z ~ eng + lnt + untr', m).fit(cov_type='cluster', cov_kwds={'groups': m.user_id})
    o['reg'] = {v: [float(fit.params[v]), [float(c) for c in fit.conf_int().loc[v]]] for v in ['eng', 'untr']}
    out[k] = o
# F-4 post hoc paraphrase check: 4/5-gram containment in own vs other students' replies
def cont(text, reps, k):
    t = tok(text); G = set(g for s in reps for g in grams(tok(s), k)); return cover(t, G, k).mean() if len(t) >= k else np.nan
pp = {}
for cl_ in ['typed', 'untraced']:
    sub = R[R.cls3 == cl_]; vals = {f'own{k}': [] for k in (4, 5)} | {f'other{k}': [] for k in (4, 5)}
    for u, s in zip(sub.user_id, sub.text):
        others = [v for v in rng.choice([x for x in users if x != u], 20, replace=False)]; orep = [r for v in others for r in REP[v]]
        for k in (4, 5): vals[f'own{k}'].append(cont(s, REP[u], k)); vals[f'other{k}'].append(cont(s, orep, k))
    pp[cl_] = {a: float(np.nanmean(b)) for a, b in vals.items()} | {'n': int(len(sub))}
out['paraphrase'] = pp
json.dump(out, open('results/r4_flora_stats.json', 'w'), indent=1); print(json.dumps(out, indent=1))
