"""Analysis log R15, F-16 to F-18 (FLoRA): stepwise decomposition of the naive-overlap estimate with paired student
bootstrap intervals; how often two replies of the embedded model to near-identical prompts share verbatim 8-grams
(a proxy for the failure of assumption A3 through a chatbot of the same family outside the environment); check of the
two counts of 118 n-grams reported in Sections 3.3 and 5.2.6.
Written after the analyses ran. No measure, class definition or parameter changes.
Usage: python -I src/r15_flora.py DATA_DIR -> results/r15_flora.json"""
import sys, re, json, time
from collections import Counter
from difflib import SequenceMatcher
import numpy as np, pandas as pd

D = sys.argv[1]; NB = 2000; rng = np.random.default_rng(2150)
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
N = 8
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
an = pd.read_pickle(f'{D}/annot.pkl')
users = sorted(set(d.user_id) & set(p.index))
P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
H = set(g for t in an.highlight_text.dropna() for g in grams(tok(t), N))
Cc = Counter(g for u in users for g in set(grams(P[u], N)))
shared = lambda g: g in H or Cc[g] - 1 >= 5
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
WG = {u: g for u, g in w.groupby('user_id')}
U1 = np.load('results/U_R1.npy'); UN = np.load('results/U_naive.npy')


def dialogue(u):
    x = d[d.user_id == u]; tbot, tpr = {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s), N): tbot.setdefault(g, pd.Timestamp(t))
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(s), N): tpr.setdefault(g, pd.Timestamp(t))
    return tbot, tpr


def editor_first(u, need):
    """first snapshot time of each n-gram of the final proposal (as Algorithm 2, line 6)"""
    g_ = WG.get(u); te = {}; need = set(need); prev = []
    if g_ is None: return te
    for t, c in zip(g_.t, g_.writing_content):
        if not need: break
        cur = tok(c); m = min(len(prev), len(cur)); pre = 0
        while pre < m and prev[pre] == cur[pre]: pre += 1
        suf = 0
        while suf < m - pre and prev[-1 - suf] == cur[-1 - suf]: suf += 1
        lo, hi = max(0, pre - N + 1), min(len(cur) - N + 1, max(len(cur) - suf, pre + 1))
        for s in range(max(lo, 0), max(hi, 0)):
            g = tuple(cur[s:s + N])
            if g in need: te[g] = t; need.discard(g)
        prev = cur
    return te


def share_C(F, lab):
    """token share covered by an n-gram labelled C, with the token priority C > E > S"""
    m = np.zeros(len(F), bool)
    for i, g in enumerate(grams(F, N)):
        if lab.get(g) == 'C': m[i:i + N] = True
    return m.mean() if len(F) else 0.0


V = {k: [] for k in ('naive', 'echo', 'echo_S', 'echo_order', 'full')}
t0 = time.time()
for k, u in enumerate(users):
    F = P[u]; G = grams(F, N); tbot, tpr = dialogue(u); te = editor_first(u, G)
    lab = {key: {} for key in V}
    for g in set(G):
        bot = g in tbot
        if not bot: continue
        pf = g in tpr and tpr[g] < tbot[g]; sh = shared(g); t_w = te.get(g)
        ef = t_w is not None and t_w < tbot[g]; ok = t_w is not None and tbot[g] <= t_w
        lab['naive'][g] = 'C'
        lab['echo'][g] = 'E' if pf else 'C'                                         # time-aware echo (R1) only
        lab['echo_S'][g] = 'E' if pf else ('S' if sh else 'C')                       # + shared source: dialogue only
        lab['echo_order'][g] = 'E' if (pf or ef) else ('C' if ok else None)          # + order check, no S
        lab['full'][g] = 'E' if (pf or ef) else ('S' if sh else ('C' if ok else None))  # U
    for key in V: V[key].append(share_C(F, lab[key]))
    if k % 50 == 0: print(k, round(time.time() - t0), flush=True)
del WG, w
V = {k: np.array(v) for k, v in V.items()}
out = {'n_students': len(users),
       'check': {'max_abs_diff_full_vs_U': float(np.abs(V['full'] - U1).max()), 'max_abs_diff_naive_vs_saved': float(np.abs(V['naive'] - UN).max())}}
ge = {k: (v >= .5).astype(float) for k, v in V.items()}


def boot(fn):
    vals = []
    for _ in range(NB):
        i = rng.integers(0, len(users), len(users)); vals.append(fn(i))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


out['F16_levels'] = {k: {'share_ge50': float(ge[k].mean()), 'ci': boot(lambda i, k=k: ge[k][i].mean()), 'median_U': float(np.median(V[k])),
                         'mean_U': float(V[k].mean())} for k in V}
steps = [('naive', 'echo', 'time-aware echo (R1)'), ('echo', 'echo_S', 'shared-source rule (R3), no order check'),
         ('echo_S', 'full', 'order check against the editor (R2, R4)'),
         ('echo', 'echo_order', 'order check without the shared-source rule'), ('echo_order', 'full', 'shared-source rule after the order check'),
         ('naive', 'full', 'all rules')]
out['F16_steps'] = []
for a, b, what in steps:
    out['F16_steps'].append({'from': a, 'to': b, 'step': what, 'diff_pp_share_ge50': float(100 * (ge[a].mean() - ge[b].mean())),
                             'ci_pp': [100 * x for x in boot(lambda i, a=a, b=b: ge[a][i].mean() - ge[b][i].mean())],
                             'students_moving_below_50': int(((ge[a] == 1) & (ge[b] == 0)).sum()),
                             'students_moving_above_50': int(((ge[a] == 0) & (ge[b] == 1)).sum()),
                             'diff_median_U_pp': float(100 * (np.median(V[a]) - np.median(V[b])))})
print(json.dumps(out['F16_steps'], indent=0), flush=True)
q = np.percentile(U1, [25, 50, 75])
out['F17_U_distribution'] = {'mean': float(U1.mean()), 'quartiles': [float(x) for x in q], 'share_zero': float((U1 == 0).mean()),
                             'deciles': [float(x) for x in np.percentile(U1, np.arange(10, 100, 10))]}

# ---------- F-18 replies of the same model to near-identical prompts (different students)
rows = []
for u, g in d.groupby('user_id'):
    for s, r in zip(g.user_utterance_text.astype(str), g.chatbot_utterance_text.astype(str)):
        pt = tok(s); rt = tok(r)
        if len(pt) >= 8 and len(rt) >= 40: rows.append((u, pt, rt))
key = lambda pt: ' '.join(pt)
byp = {}
for i, (u, pt, rt) in enumerate(rows): byp.setdefault(key(pt), []).append(i)
pairs = []
for kk, idx in byp.items():
    for a in idx:
        for b in idx:
            if a < b and rows[a][0] != rows[b][0]: pairs.append((a, b, 'identical'))
# near-identical: 8-gram Jaccard >= 0.8 between prompts of different students (prompts of 8-60 tokens)
pg = [set(grams(pt, N)) if 8 <= len(pt) <= 60 else set() for _, pt, _ in rows]
inv = {}
for i, s in enumerate(pg):
    for g in s: inv.setdefault(g, set()).add(i)
seen = {(a, b) for a, b, _ in pairs}
for i, s in enumerate(pg):
    if not s: continue
    cand = Counter(j for g in s for j in inv[g] if j > i)
    for j, c in cand.items():
        if rows[i][0] == rows[j][0] or (i, j) in seen: continue
        jac = c / len(s | pg[j])
        if jac >= .8: pairs.append((i, j, 'near')); seen.add((i, j))


def cover(rt_a, rt_b, n=N):
    ga = set(grams(rt_a, n)); m = np.zeros(len(rt_b), bool)
    for i, g in enumerate(grams(rt_b, n)):
        if g in ga: m[i:i + n] = True
    return m.mean()


cov = np.array([max(cover(rows[a][2], rows[b][2]), cover(rows[b][2], rows[a][2])) for a, b, _ in pairs]) if pairs else np.zeros(0)
# random pairs of replies of different students as reference
rr = []
for _ in range(5000):
    a, b = rng.integers(0, len(rows), 2)
    if rows[a][0] != rows[b][0]: rr.append(max(cover(rows[a][2], rows[b][2]), cover(rows[b][2], rows[a][2])))
rr = np.array(rr)
out['F18_same_prompt_replies'] = {
    'prompt_reply_pairs_considered': len(rows), 'pairs_identical_prompt': sum(1 for *_, k in pairs if k == 'identical'),
    'pairs_near_identical_prompt': sum(1 for *_, k in pairs if k == 'near'), 'n_pairs': len(pairs),
    'students_in_pairs': len({rows[a][0] for a, b, _ in pairs} | {rows[b][0] for a, b, _ in pairs}),
    'coverage_of_one_reply_by_8grams_of_the_other': {'mean': float(cov.mean()) if len(cov) else None, 'median': float(np.median(cov)) if len(cov) else None,
                                                    'p90': float(np.percentile(cov, 90)) if len(cov) else None, 'share_pairs_gt_0': float((cov > 0).mean()) if len(cov) else None,
                                                    'share_pairs_ge_25': float((cov >= .25).mean()) if len(cov) else None,
                                                    'ci_mean': None},
    'random_reply_pairs': {'n': len(rr), 'mean': float(rr.mean()), 'median': float(np.median(rr)), 'share_gt_0': float((rr > 0).mean()), 'p90': float(np.percentile(rr, 90))}}
if len(cov):
    vals = [cov[rng.integers(0, len(cov), len(cov))].mean() for _ in range(NB)]
    out['F18_same_prompt_replies']['coverage_of_one_reply_by_8grams_of_the_other']['ci_mean'] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
print(json.dumps(out['F18_same_prompt_replies'], indent=0), flush=True)

# ---------- check of the two "118" counts (Sections 3.3 and 5.2.6)
r11 = json.load(open('results/r11_flora.json')); r13 = json.load(open('results/r13_flora.json'))
def find(o, pat, path=''):
    res = []
    if isinstance(o, dict):
        for k, v in o.items(): res += find(v, pat, f'{path}/{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): res += find(v, pat, f'{path}[{i}]')
    elif o == pat: res.append(path)
    return res
out['check_118'] = {'r11_flora_paths_equal_118': find(r11, 118), 'r13_flora_paths_equal_118': find(r13, 118),
                    'r13_X0_8grams_total': r13['F14_first_snapshot']['X0_8grams_total']}
out['seconds'] = time.time() - t0
json.dump(out, open('results/r15_flora.json', 'w'), indent=1, default=float)
print('done', out['seconds'])
