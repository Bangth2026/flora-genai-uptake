"""Analysis log R16, F-19 to F-21 (FLoRA).
F-19 the attribution recomputed from the raw logs with the R16 feature code, checked against the released token labels;
F-20 subtypes of the unresolved class with student-bootstrap intervals, and the share of each subtype inside logged
pastes; F-21 the evidence-fusion classifiers of B-26 (trained on simulated students only) applied to the 275 proposals,
compared with U and with the logged pastes that neither method reads.
Written before these analyses ran; see ANALYSIS_LOG_R16.md. No measure or parameter of the attribution changes.
Usage: python -I src/r16_flora.py DATA_DIR R8_TOKEN_CLASSES -> results/r16_flora.json
DATA_DIR: dialogue.xlsx, proposal.xlsx, writing.pkl, annot.pkl, trace.pkl (Figshare 5.1-5.4, 5.8, 5.2 as one table)."""
import sys, os, json, time, pickle
from collections import Counter
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r16_features as F
src = open(os.path.join(HERE, 'r11_benchmark.py')).read().replace("'results/flora_token_classes.csv.gz'", repr(sys.argv[2]))
defs = src[:src.index("out = {'n_per_seed': 400}")]
D = sys.argv[1]; sys.argv = [os.path.join(HERE, 'r11_benchmark.py'), D]
G_ = {'__name__': 'r11_defs', '__file__': os.path.join(HERE, 'r11_benchmark.py')}
exec(compile(defs, 'r11_benchmark.py', 'exec'), G_)
attribute, tokens, ngrams, first_times, N, TAU = (G_[k] for k in ('attribute', 'tokens', 'ngrams', 'first_times', 'N', 'TAU'))
t00 = time.perf_counter(); NB = 2000; rng = np.random.default_rng(2170)
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)

d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
an = pd.read_pickle(f'{D}/annot.pkl')
w = pd.read_pickle(f'{D}/writing.pkl').sort_values(['user_id', 'writing_log_time'])
tr = pd.read_pickle(f'{D}/trace.pkl')
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); del tr
pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
users = sorted(set(d.user_id) & set(p.index))
sec = lambda x: pd.Timestamp(x).value / 1e9
P = {u: tokens(p.loc[u, 'final_proposal']) for u in users}
H = {g for t in an.highlight_text.dropna() for g in ngrams(tokens(t), N)}
Cc = Counter(g for u in users for g in set(ngrams(P[u], N)))
G_['SHG'] = H | {g for g, c in Cc.items() if c - 1 >= 5}   # reading highlights and 8-grams in >= 5 other proposals
WG = {u: g for u, g in w.groupby('user_id')}; DG = {u: g for u, g in d.groupby('user_id')}
ST = {}
for u in users:
    x = DG[u]; g = WG.get(u)
    ST[u] = dict(final=str(p.loc[u, 'final_proposal']), donor=u,
                 replies=sorted((sec(t), str(s)) for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text)),
                 turns=sorted((sec(t), str(s)) for t, s in zip(x.user_ask_time, x.user_utterance_text)),
                 snaps=[] if g is None else [(float(t) / 1e3, str(c)) for t, c in zip(g.writing_log_time, g.writing_content)])
log('loaded', len(users))

# cohort: replies given to other students
OWN = {}
for u in users:
    for _, s in ST[u]['replies']:
        for g in set(ngrams(tokens(s), N)): OWN.setdefault(g, set()).add(u)
ENC = F.Encoder(); PT, PU = [], []
for u in users:
    for _, s in ST[u]['replies']:
        tk = tokens(s)
        for a, b in F.windows(tk): PT.append(' '.join(tk[a:b])); PU.append(u)
PU = np.array(PU); PE = ENC(PT); ENC.save(); log('cohort pool', len(PT))

class Fuzzy(dict):
    """tolerant matching of R15 B-20: n-grams matched exactly or with one token replaced"""
    def __init__(self, items, n):
        super().__init__()
        for t, text in sorted(items, key=lambda x: x[0]):
            for g in ngrams(tokens(text), n):
                for key in [g] + [g[:i] + ('\0',) + g[i + 1:] for i in range(n)]: dict.setdefault(self, key, t)
    def _keys(self, g): return [g] + [g[:i] + ('\0',) + g[i + 1:] for i in range(len(g))]
    def __contains__(self, g): return any(dict.__contains__(self, k) for k in self._keys(g))
    def __getitem__(self, g): return min(dict.__getitem__(self, k) for k in self._keys(g) if dict.__contains__(self, k))
def attribute_fuzzy(st):
    orig = G_['first_times']; G_['first_times'] = lambda items, n: Fuzzy(items, n)
    try: return attribute(st, st['snaps'], entry_rule='token')
    finally: G_['first_times'] = orig


# ---------- F-19 recompute the attribution and features
M = pickle.load(open('results/r16_fusion_models.pkl', 'rb')); TH_OWN, TH_COH = M['theta_own'], M['theta_cohort']
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])
REL = {u: g['class'].tolist() for u, g in tc.groupby('user_id')}
EV = {}; agree_n = agree_d = 0
for k, u in enumerate(users):
    st = ST[u]
    if not st['snaps']: continue
    A = tokens(st['final']); o = attribute(st, st['snaps'], entry_rule='token', keep=True)
    s_any, s_dir = F.semantic(A, st['replies'], st['turns'], tokens, ENC)
    s_coh = F.cohort_max(A, ENC, PE, PU == u)
    cg = lambda g, u=u: bool(OWN.get(g, set()) - {u})
    fz = [c == 'C' for c in attribute_fuzzy(st)['cls']]
    X, aux = F.features(st, o, N, TAU, tokens, ngrams, fz, s_any, s_dir, cg, s_coh)
    EV[u] = dict(A=A, cls=o['cls'], X=X, aux=aux, fz=fz)
    if u in REL and len(REL[u]) == len(A): agree_n += sum(a == b for a, b in zip(REL[u], o['cls'])); agree_d += len(A)
    if k % 25 == 0: log('student', k)
out = {'F19_students': len(EV), 'F19_agreement_with_released_labels': agree_n / agree_d if agree_d else None,
       'F19_tolerant_C_mean_share': None}
out['F19_tolerant_C_mean_share'] = float(np.mean([np.mean(e['fz']) for e in EV.values()]))
log('F19 agreement', out['F19_agreement_with_released_labels'])

# pasted text of the 68 students with logged pastes
PG = {u: {g for t in x.txt for g in ngrams(tokens(t), N)} for u, x in pa.groupby('user_id') if u in EV}
def in_paste(u):
    A = EV[u]['A']; m = np.zeros(len(A), bool)
    for i, g in enumerate(ngrams(A, N)):
        if g in PG[u]: m[i:i + N] = True
    return m
INP = {u: in_paste(u) for u in PG}


def boot_mean(vals):
    v = np.array(vals, float); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def boot_ratio(num, den):
    a, b = np.array(num, float), np.array(den, float); v = []
    for _ in range(NB):
        i = rng.integers(0, len(a), len(a)); v.append(a[i].sum() / max(b[i].sum(), 1))
    return [float(a.sum() / max(b.sum(), 1)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ---------- F-20 subtypes of X
SUB = {u: F.x_subtypes(e['cls'], e['aux'], TH_OWN, TH_COH) for u, e in EV.items()}
o20 = {'theta_own': TH_OWN, 'theta_cohort': TH_COH, 'mean_share_of_proposal': {}, 'pooled_share_of_X': {}, 'in_logged_pastes': {},
       'students_with_subtype': {}}
nX = sum(sum(s is not None for s in v) for v in SUB.values())
for s in F.XSUB:
    o20['mean_share_of_proposal'][s] = boot_mean([np.mean([x == s for x in SUB[u]]) for u in EV])
    o20['pooled_share_of_X'][s] = sum(sum(x == s for x in v) for v in SUB.values()) / nX
    o20['students_with_subtype'][s] = int(sum(any(x == s for x in SUB[u]) for u in EV))
    num = [int(((np.array(SUB[u]) == s) & INP[u]).sum()) for u in INP]; den = [int((np.array(SUB[u]) == s).sum()) for u in INP]
    o20['in_logged_pastes'][s] = boot_ratio(num, den) + [int(sum(den))]
# each condition on its own (not exclusive), pooled over X tokens, and the overlap of the semantic flag with first-snapshot text
cond = Counter(); both = Counter(); nXa = 0
for u, e in EV.items():
    a = e['aux']; first = {}
    for j, c in enumerate(e['cls']):
        if c != 'X': continue
        nXa += 1
        f = {'log_gap': a['io'][j] is None, 'reentered': bool(a['reent'][j]), 'same_segment_as_C': a['segC'][j],
             'same_snapshot_as_C': a['sid'][j] is not None and a['sid'][j] in {a['sid'][k] for k, cc in enumerate(e['cls']) if cc == 'C' and a['sid'][k] is not None},
             'near_own_reply': a['s_dir'][j] >= TH_OWN, 'cohort_reply': a['incoh'][j] > 0 or a['s_cohort'][j] >= TH_COH,
             'first_snapshot': a['sid'][j] == a['first_nonempty']}
        for k_, v in f.items(): cond[k_] += v
        if f['near_own_reply']:
            both['near_own_and_first_snapshot'] += f['first_snapshot']; both['near_own_and_same_snapshot_as_C'] += f['same_snapshot_as_C']
o20['conditions_alone_share_of_X'] = {k: v / nXa for k, v in cond.items()}
o20['near_own_reply_overlap_share_of_flagged'] = {k: v / max(cond['near_own_reply'], 1) for k, v in both.items()}
o20['mean_share_X'] = boot_mean([np.mean([c == 'X' for c in EV[u]['cls']]) for u in EV])
out['F20_x_subtypes'] = o20; log('F20', {k: round(v[0], 4) for k, v in o20['mean_share_of_proposal'].items()})

# ---------- F-21 fusion classifiers trained on simulated students, applied to FLoRA
o21 = {}
U = np.array([np.mean([c == 'C' for c in EV[u]['cls']]) for u in EV])
for mk in ('gbt_train_base', 'gbt_train_all'):
    mdl = M[mk]; shares = {c: [] for c in F.CLASSES}; fC = {}; pC = {}
    for u, e in EV.items():
        Pp = mdl.predict_proba(e['X']); pred = Pp.argmax(1); fC[u] = pred == 0; pC[u] = Pp[:, 0]
        for k, c in enumerate(F.CLASSES): shares[c].append(float((pred == k).mean()))
    Uf = np.array(shares['C']); rule = {u: np.array(e['cls']) for u, e in EV.items()}
    from scipy import stats
    r = {'mean_share': {c: boot_mean(v) for c, v in shares.items()},
         'median_C': float(np.median(Uf)), 'share_students_ge50_fusion': boot_mean(list(Uf >= .5)), 'share_students_ge50_U': boot_mean(list(U >= .5)),
         'spearman_with_U': float(stats.spearmanr(U, Uf)[0]),
         'cross_tab_rule_by_fusion_pooled': {}}
    ct = Counter()
    for u, e in EV.items():
        pred = mdl.predict(e['X'])
        for a, b in zip(rule[u], pred): ct[(a, F.CLASSES[b])] += 1
    r['cross_tab_rule_by_fusion_pooled'] = {f'{a}->{b}': v for (a, b), v in sorted(ct.items())}
    # logged pastes: share of fusion-C tokens inside pasted text, overall and for tokens the rules leave in X
    r['fusion_C_in_logged_pastes'] = boot_ratio([int((fC[u] & INP[u]).sum()) for u in INP], [int(fC[u].sum()) for u in INP])
    r['rule_C_in_logged_pastes'] = boot_ratio([int(((rule[u] == 'C') & INP[u]).sum()) for u in INP], [int((rule[u] == 'C').sum()) for u in INP])
    r['fusion_C_rule_X_in_logged_pastes'] = boot_ratio([int((fC[u] & (rule[u] == 'X') & INP[u]).sum()) for u in INP],
                                                      [int((fC[u] & (rule[u] == 'X')).sum()) for u in INP])
    r['rule_X_in_logged_pastes'] = boot_ratio([int(((rule[u] == 'X') & INP[u]).sum()) for u in INP], [int((rule[u] == 'X').sum()) for u in INP])
    # where the fusion C tokens that the rules leave in X sit among the X subtypes
    sub_ct = Counter(s for u in EV for s, f in zip(SUB[u], fC[u]) if s is not None and f)
    r['fusion_C_within_X_by_subtype'] = {s: sub_ct[s] / max(sum(sub_ct.values()), 1) for s in F.XSUB}
    r['fusion_C_on_rule_X_tokens_share_of_X'] = sum(sub_ct.values()) / nX
    o21[mk] = r; log('F21', mk, round(float(np.mean(Uf)), 3), r['fusion_C_in_logged_pastes'])
out['F21_fusion_on_flora'] = o21
out['seconds'] = time.perf_counter() - t00
json.dump(out, open('results/r16_flora.json', 'w'), indent=1, default=float)
log('done')
