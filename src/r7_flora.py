"""Plan R7, FLoRA (PREREG_R7.md H1): students with and without logged pastes. Data readers, the snapshot scan and the
labelling rules are copied from src/r6_flora.py (which reproduces U* exactly).
Usage: python -I src/r7_flora.py DATA_DIR -> results/r7_flora.json"""
import sys, re, json, warnings
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2060); NB = 2000
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
an = pd.read_pickle(f'{D}/annot.pkl'); tr = pd.read_pickle(f'{D}/trace.pkl')
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); del tr
pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms'); pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
users = sorted(set(d.user_id) & set(p.index))
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
users_all = users; users = users[:LIMIT] if LIMIT else users
score = p.loc[users, 'proposal_score'].values.astype(float)
NS = (8,)
H = {n: set(g for t in an.highlight_text.dropna() for g in grams(tok(t), n)) for n in NS}
Cc = {n: Counter(g for u in users_all for g in set(grams(P[u], n))) for n in NS}
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
WG = {u: g for u, g in w.groupby('user_id')}
U1 = np.load('results/U_R1.npy')[:len(users)]
THIN = (5, 15, 60, 300); TAUS = (1, 2, 5, 10, 20, 50)

def dialogue(u, n):
    x = d[d.user_id == u]; tbot, tpr, plen = {}, {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s), n): tbot.setdefault(g, pd.Timestamp(t))
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(s), n): tpr.setdefault(g, pd.Timestamp(t))
    return tbot, tpr

def scan(enc, times, idx, n, need, want_seen=False):
    """Entry of each n-gram of A (int-encoded tuples in need): time, insertion count of the whole transition,
    size of the inserted segment containing it, transition index. Optionally first-seen time of every n-gram."""
    need = set(need); ted, ins, seg, tid, seen = {}, {}, {}, {}, {}
    prev = np.zeros(0, np.int32)
    for k, i in enumerate(idx):
        if not need and not want_seen: break
        cur = enc[i]; L = len(cur)
        if len(prev) == 0:
            segs = [(0, L)] if L else []; lo, hi = 0, L - n + 1
        else:
            m = min(len(prev), L); ne = np.flatnonzero(prev[:m] != cur[:m]); pre = int(ne[0]) if len(ne) else m
            mm = min(len(prev), L) - pre; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
            a, b = prev[pre:len(prev) - suf], cur[pre:L - suf]
            if len(a) == 0 and len(b) == 0: prev = cur; continue
            if len(b) == 0: segs = []
            elif len(a) == 0 or len(b) == 1: segs = [(pre, pre + len(b))]
            else:
                segs = [(pre + j1, pre + j2) for op, i1, i2, j1, j2 in SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes() if op in ('insert', 'replace')]
            lo, hi = max(0, pre - n + 1), min(L - n + 1, max(L - suf, pre + 1))
        tot = sum(j2 - j1 for j1, j2 in segs); t = times[i]
        for s in range(max(lo, 0), max(hi, 0)):
            g = tuple(cur[s:s + n].tolist())
            if want_seen and g not in seen: seen[g] = t
            if g in need:
                best = (0, 0)
                for j1, j2 in segs:
                    ov = min(s + n, j2) - max(s, j1)
                    if ov > best[0]: best = (ov, j2 - j1)
                ted[g] = t; ins[g] = tot; seg[g] = best[1]; tid[g] = k; need.discard(g)
        prev = cur
    return dict(ted=ted, ins=ins, seg=seg, tid=tid, seen=seen)

def classify(FGs, tbot, tpr, sc, n, tau=5, delta=pd.Timedelta(0), mode='ins', k_peer=5):
    cls = {}
    for g, gi in FGs.items():
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]; shared = g in H[n] or Cc[n][g] - 1 >= k_peer
        te = sc['ted'].get(gi)
        if bot and not echo and not shared and te is not None and tbot[g] <= te + delta: cls[g] = 'C'
        elif bot and (echo or (te is not None and te < tbot[g])): cls[g] = 'E'
        elif shared: cls[g] = 'S'
        elif te is None: cls[g] = 'X'
        else: cls[g] = 'T' if sc[mode][gi] <= tau else 'X'
    return cls

def project(F, n, cls, order='CESTX'):
    rank = {c: i for i, c in enumerate(order)}; best = np.full(len(F), 99)
    for i, g in enumerate(grams(F, n)): best[i:i + n] = np.minimum(best[i:i + n], rank[cls[g]])
    return np.array([order[b] if b < 99 else 'X' for b in best])

def majority(F, n, cls, order='CESTX'):
    votes = [Counter() for _ in F]; rank = {c: i for i, c in enumerate(order)}
    for i, g in enumerate(grams(F, n)):
        for j in range(i, i + n): votes[j][cls[g]] += 1
    return np.array([min(v, key=lambda c: (-v[c], rank[c])) if v else 'X' for v in votes])

bio = pd.read_excel(f'{D}/bio.xlsx').drop_duplicates('user_id').set_index('user_id')
eng = bio.reindex(users).mother_tongue.astype(str).str.strip().str.lower().eq('english').astype(float).where(bio.reindex(users).mother_tongue.notna())
cls_sh = pd.read_csv('results/r5_flora_classes.csv', index_col=0).reindex(users)
pu = set(pa.user_id)
rows = []
for k, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in (g_.writing_content if g_ is not None else [])]
    F = P[u]; Fi = [vocab.setdefault(x, len(vocab)) for x in F]
    tbot, tpr = dialogue(u, 8); FG = grams(F, 8); FGs = {g: tuple(Fi[i:i + 8]) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), 8, set(FGs.values()))
    cls = classify(FGs, tbot, tpr, sc, 8)
    # transitions: number inserting > 20 tokens (whole-transition count), and how many of them carried a C 8-gram
    nlarge = 0; prev = np.zeros(0, np.int32)
    for cur in enc:
        m = min(len(prev), len(cur)); ne = np.flatnonzero(prev[:m] != cur[:m]); pre = int(ne[0]) if len(ne) else m
        mm = m - pre; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
        a, b = prev[pre:len(prev) - suf], cur[pre:len(cur) - suf]
        if len(b) > 20:
            ins = len(b) if len(a) == 0 else sum(j2 - j1 for op, i1, i2, j1, j2 in SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes() if op in ('insert', 'replace'))
            nlarge += ins > 20
        prev = cur
    large_with_C = len({sc['tid'][FGs[g]] for g, c in cls.items() if c == 'C' and FGs[g] in sc['tid'] and sc['ins'][FGs[g]] > 20})
    x = d[d.user_id == u]
    rows.append(dict(u=u, logged=u in pu, large=nlarge, large_with_C=large_with_C, length=len(F),
                     bot_words=int(sum(len(str(s).split()) for s in x.chatbot_utterance_text)), turns=len(x),
                     grade=float(p.loc[u, 'proposal_score']), english=eng.loc[u], U=float((project(F, 8, cls) == 'C').mean()),
                     **{c: float(cls_sh.loc[u, c]) for c in 'CETX'}))
    if k % 50 == 0: print(k, flush=True)
R = pd.DataFrame(rows).set_index('u')
out = {'check_U_max_abs_diff': float(np.abs(R.U.values - U1).max()), 'n_logged': int(R.logged.sum()), 'n_unlogged': int((~R.logged).sum())}
rng = np.random.default_rng(2070)
A_, B_ = R[R.logged], R[~R.logged]
def boot_diff(a, b, f):
    v = [f(rng.choice(a, len(a))) - f(rng.choice(b, len(b))) for _ in range(NB)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
comp = {}
for col, f, nm in [('large', np.median, 'median'), ('length', np.median, 'median'), ('bot_words', np.median, 'median'), ('turns', np.median, 'median'),
                   ('grade', np.mean, 'mean'), ('U', np.median, 'median'), ('C', np.mean, 'mean'), ('E', np.mean, 'mean'), ('T', np.mean, 'mean'), ('X', np.mean, 'mean')]:
    a, b = A_[col].values.astype(float), B_[col].values.astype(float)
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    comp[col] = {'stat': nm, 'logged': float(f(a)), 'unlogged': float(f(b)), 'diff': float(f(a) - f(b)), 'diff_ci': boot_diff(a, b, f),
                 'mannwhitney_p': float(stats.mannwhitneyu(a, b).pvalue), 'smd': float((a.mean() - b.mean()) / sd) if sd > 0 else None}
e_a, e_b = A_.english.dropna(), B_.english.dropna()
ct = [[int(e_a.sum()), int(len(e_a) - e_a.sum())], [int(e_b.sum()), int(len(e_b) - e_b.sum())]]
comp['english'] = {'logged': float(e_a.mean()), 'unlogged': float(e_b.mean()), 'n_known': [int(len(e_a)), int(len(e_b))],
                   'chi2_p': float(stats.chi2_contingency(ct)[1]), 'fisher_p': float(stats.fisher_exact(ct)[1])}
for grp, G in [('logged', A_), ('unlogged', B_)]:
    comp.setdefault('large_insertions_with_C', {})[grp] = float(G.large_with_C.sum() / max(1, G.large.sum()))
out['H1'] = comp
json.dump(out, open('results/r7_flora.json' if not LIMIT else '/tmp/r7_test.json', 'w'), indent=1); print(json.dumps(out, indent=1))
