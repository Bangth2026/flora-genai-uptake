"""Plan R3, part F (post hoc): five provenance classes on FLoRA; detectors on typed vs untraced paragraphs.
Usage: python -I src/run_r3_flora.py DATA_DIR -> results/r3_flora_stats.json, results/r3_flora_paragraphs.pkl"""
import sys, re, json, warnings
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2031); NB = 2000; NET = 5
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, n: [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
an = pd.read_pickle(f'{D}/annot.pkl'); tr = pd.read_pickle(f'{D}/trace.pkl')
users = sorted(set(d.user_id) & set(p.index)); P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
H = set(g for t in an.highlight_text.dropna() for g in grams(tok(t), 8)); Cc = Counter(g for u in users for g in set(grams(P[u], 8)))
has_trace = set(tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].user_id)
RANK = {'C': 0, 'E': 1, 'S': 2, 'T': 3, 'X': 4}; INV = {v: k for k, v in RANK.items()}

def classes(u):
    x = d[d.user_id == u]; FG = grams(P[u], 8); need = set(FG); tbot, tpr = {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s), 8): tbot.setdefault(g, pd.Timestamp(t))
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(s), 8): tpr.setdefault(g, pd.Timestamp(t))
    ted, mode, prev = {}, {}, 0
    for t, c in zip(w[w.user_id == u].t, w[w.user_id == u].writing_content):
        tk = tok(c); ch = abs(len(tk) - prev); prev = len(tk)
        if not need: continue
        for g in grams(tk, 8):
            if g in need: ted[g] = t; mode[g] = 'T' if ch <= NET else 'X'; need.discard(g)
    cls = {}
    for g in set(FG):
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]; shared = g in H or Cc[g] - 1 >= 5
        if bot and not echo and not shared and g in ted and tbot[g] <= ted[g]: cls[g] = 'C'
        elif bot and (echo or (g in ted and ted[g] < tbot[g])): cls[g] = 'E'
        elif shared: cls[g] = 'S'
        else: cls[g] = mode.get(g, 'X')
    best = np.full(len(P[u]), 9)
    for i, g in enumerate(FG): best[i:i + 8] = np.minimum(best[i:i + 8], RANK[cls[g]])
    return np.array([INV.get(b, 'X') for b in best]), cls

LAB = {}
for k, u in enumerate(users):
    LAB[u] = classes(u)
    if k % 25 == 0: print(k, flush=True)
sh = pd.DataFrame({c: [float((LAB[u][0] == c).mean()) if len(P[u]) else 0 for u in users] for c in 'CESTX'}, index=users)
U1 = np.load('results/U_R1.npy')
out = {'class_means': {c: float(sh[c].mean()) for c in 'CESTX'}, 'class_medians': {c: float(sh[c].median()) for c in 'CESTX'},
       'C_equals_Ustar_maxdiff': float(np.abs(sh.C.values - U1).max()),
       'by_trace': {k: {c: float(sh.loc[[u for u in users if (u in has_trace) == v], c].mean()) for c in 'CESTX'} | {'n': int(sum((u in has_trace) == v for u in users))}
                    for k, v in [('with_trace', True), ('without_trace', False)]},
       'non_C_share_X': float((sh.X / (1 - sh.C)).replace([np.inf], np.nan).mean())}

# paragraph-level: R2 paragraphs with token classes
R = pd.read_pickle('results/r2_scores.pkl')
def para_cls(u, text):
    t = tok(text); F = P[u]; n = len(t)
    for i in range(len(F) - n + 1):
        if F[i:i + n] == t: return LAB[u][0][i:i + n]
    return None
pc = [para_cls(u, s) for u, s in zip(R.user_id, R.text)]
R['found'] = [c is not None for c in pc]
for c in 'TX': R[c] = [float((a == c).mean()) if a is not None else np.nan for a in pc]
comp = R[R.label == 'composed']
R['cls3'] = np.where(R.label == 'chatbot', 'chatbot', np.where((R.label == 'composed') & (R['T'] >= .9), 'typed', np.where((R.label == 'composed') & (R['X'] >= .9), 'untraced', 'other')))
out['paragraphs'] = {'composed_n': int(len(comp)), 'found': float(comp.found.mean()), 'counts': R.cls3.value_counts().to_dict(),
                     'composed_typed_share': float((comp['T'] >= .9).mean()), 'composed_untraced_share': float((comp['X'] >= .9).mean()),
                     'typed_by_l1': {k: float((comp[comp.eng == v]['T'] >= .9).mean()) for k, v in [('eng', 1), ('other', 0)]},
                     'typed_students_eng': int(R[(R.cls3 == 'typed') & (R.eng == 1)].user_id.nunique()),
                     'typed_paras_eng': int(((R.cls3 == 'typed') & (R.eng == 1)).sum()),
                     'typed_students_total': int(R[R.cls3 == 'typed'].user_id.nunique())}
DET = ['D1_ll', 'D2_fdg', 'D3_bino']; R['D3_bino'] = -R['D3_bino']
studs = R.user_id.unique(); idx = {u: np.where(R.user_id.values == u)[0] for u in studs}
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
for k in DET:
    o = {}
    for a, b in [('chatbot', 'typed'), ('chatbot', 'untraced'), ('untraced', 'typed')]:
        o[f'{a}_vs_{b}'] = [float(auc(R, a, b, k)), cb(lambda s_: auc(s_, a, b, k))]
    sd = R[R.label == 'composed'][k].std()
    for grp in ['typed', 'untraced']:
        f = lambda s_: (s_[(s_.cls3 == grp) & (s_.eng == 0)][k].mean() - s_[(s_.cls3 == grp) & (s_.eng == 1)][k].mean()) / sd
        o[f'l1_diff_sd_{grp}'] = [float(f(R)), cb(f)]
    m = R[R.cls3.isin(['typed', 'untraced'])].copy(); m['z'] = (m[k] - m[k].mean()) / m[k].std(); m['untr'] = (m.cls3 == 'untraced').astype(int); m['lnt'] = np.log(m.ntok)
    fit = smf.ols('z ~ eng + lnt + untr', m.dropna(subset=['eng'])).fit(cov_type='cluster', cov_kwds={'groups': m.dropna(subset=['eng']).user_id})
    o['reg'] = {v: [float(fit.params[v]), [float(c) for c in fit.conf_int().loc[v]]] for v in ['eng', 'untr', 'lnt']}
    th80 = np.percentile(R[R.cls3 == 'chatbot'][k], 20)
    o['flag80_typed'] = {e: float((R[(R.cls3 == 'typed') & (R.eng == v)][k] >= th80).mean()) for e, v in [('eng', 1), ('other', 0)]}
    o['flag80_untraced'] = float((R[R.cls3 == 'untraced'][k] >= th80).mean())
    out[k] = o
R.drop(columns=['text']).to_pickle('results/r3_flora_paragraphs.pkl')
sh.to_csv('results/r3_flora_classes.csv'); json.dump(out, open('results/r3_flora_stats.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
