"""Analysis log R17, B-29 to B-34 (benchmark).
B-29 risk--coverage in the standard form (risk over decided tokens only; El-Yaniv & Wiener, 2010) for every method
with a graded score, at a common coverage, on the four test sets of R15; B-30 where the attribution and earliest
plausible attribution with abstention by entry size differ: paired student-bootstrap intervals for the difference in
the standard-form area, on all routes and with the shared-source route left out, and the difference with each route
left out; B-31 the assumption-A3 set: false chatbot labels of each method and what fills the decided set at the common
coverage; B-32 a hyperparameter search for the evidence-fusion trees on the validation set (seed 2700); B-33 the
simulator with other outside text drawn from a different source: (a) the unresolved FLoRA runs that carry no subtype
evidence (X-no-evidence, written by src/r17_flora_xsub.py), (b) arXiv abstracts in statistics and computing; the four
test sets are rebuilt with the same seeds and only the text of the outside routes changes; B-34 the students passed to
the LLM baseline (src/r17_llm_baseline.py).
The students of seeds 2100--2700 are rebuilt exactly as in R16. No measure, class definition, parameter or code of the
attribution changes. Written before these analyses ran; see ANALYSIS_LOG_R17.md.
Usage: EMB_CACHE=FILE python -I src/r17_benchmark.py DATA_DIR R8_TOKEN_CLASSES XNOEV_CSV ARXIV_JSON WORK_DIR
-> results/r17_benchmark.json, results/r17_fusion_tuned.pkl, WORK_DIR/r17_llm_students.pkl"""
import sys, os, json, random, time, pickle
from collections import Counter
from difflib import SequenceMatcher
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r16_features as F
XNOEV, ARXIV, WORK = sys.argv[3], sys.argv[4], sys.argv[5]
r15 = open(os.path.join(HERE, 'r15_benchmark.py')).read()
NEW = r15[r15.index('NEW = """') + 9: r15.index('"""\nanchor')] + '\n'
src = open(os.path.join(HERE, 'r11_benchmark.py')).read()
src = src.replace("'results/flora_token_classes.csv.gz'", repr(sys.argv[2]))
cut = src.index("out = {'n_per_seed': 400}")
anchor = "        elif r.startswith('E'):"
defs = src[:cut].replace(anchor, NEW + anchor)
defs = defs.replace("TRUTHMAP = {'M': 'T',", "TRUTHMAP = {'A3-after': 'X', 'A3-before': 'X', 'A3-regen': 'X', 'M': 'T',")
DATA_DIR = sys.argv[1]
sys.argv = [os.path.join(HERE, 'r11_benchmark.py'), DATA_DIR]
G_ = {'__name__': 'r11_defs', '__file__': os.path.join(HERE, 'r11_benchmark.py')}
exec(compile(defs, 'r11_benchmark.py', 'exec'), G_)
# selective-prediction code of R12 (token entry sizes, every baseline with a graded score), run in the same namespace
r12 = open(os.path.join(HERE, 'r12_benchmark.py')).read()
sel_src = r12[r12.index('def token_iota'): r12.index('class OnlineR11')]
sel_src = sel_src.replace("    out = {k: rc_summary(v) for k, v in rc.items()}", "    globals()['_RC'] = rc; globals()['_PER'] = per\n    out = {k: rc_summary(v) for k, v in rc.items()}")
G_['Q'] = 1; G_['SequenceMatcher'] = SequenceMatcher   # Q=1 only shortens R12's paper-convention bootstrap, not reported here
exec(compile(sel_src, 'r12_benchmark.py', 'exec'), G_)
for k in ('make_student', 'BASE', 'EXT', 'attribute', 'scan', 'first_times', 'tokens', 'ngrams', 'N', 'TAU', 'ORDER',
          'counts', 'prf_from', 'bin_counts', 'bin_prf', 'student_err', 'REPL', 'SHARED_TXT', 'XRUN', 'selective'):
    globals()[k] = G_[k]
Q = int(os.environ.get('QUICK', 0)); NS = 40 if Q else 400; NB = 200 if Q else 1000; NB30 = 20 if Q else 300
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)

HARDX = BASE + [('X-retype', .05), ('X-chunks', .05), ('X-mixed', .05)]
A3 = BASE + [('A3-after', .05), ('A3-before', .05), ('A3-regen', .05)]
ALL = EXT + HARDX[-3:] + A3[-3:]
SPEC = [('base', 2100, BASE), ('ext', 2200, EXT), ('hard_x', 2300, HARDX), ('a3', 2400, A3),
        ('train_base', 2500, BASE), ('train_all', 2600, ALL), ('val', 2700, BASE)]
def build(names, xrun=None):
    orig = G_['XRUN']
    if xrun is not None: G_['XRUN'] = xrun
    try:
        S_ = {}
        for name, seed, routes in SPEC:
            if name in names: R = random.Random(seed); S_[name] = [make_student(R, routes) for _ in range(NS)]
        return S_
    finally: G_['XRUN'] = orig
SETS = build([s[0] for s in SPEC]); TEST = ['base', 'ext', 'hard_x', 'a3']
log('students built')
out = {'n_per_set': NS}

# ---------- B-34 students for the LLM baseline (first 20 of three test sets)
os.makedirs(WORK, exist_ok=True)
pickle.dump({k: SETS[k][:20] for k in ('base', 'ext', 'a3')}, open(f'{WORK}/r17_llm_students.pkl', 'wb'))


# ---------- B-29 standard-form risk--coverage for every graded method
def rc_standard(rows, c_common=None, curve=False):
    dec = sorted([(a, b, c) for a, b, c in rows if c != -1], key=lambda x: -x[2]); n_all = len(rows); n_ = len(dec)
    if not n_: return {'coverage': 0.0}
    err = np.cumsum([a != b for a, b, _ in dec]) / np.arange(1, n_ + 1); cov = np.arange(1, n_ + 1) / n_all
    o = {'coverage': float(cov[-1]), 'selective_risk': float(err[-1]), 'aurc_own_range': float(np.trapezoid(err, cov) / cov[-1])}
    if c_common:
        k = max(1, int(np.searchsorted(cov, c_common, side='right'))); o['aurc_common_range'] = float(np.trapezoid(err[:k], cov[:k]) / cov[k - 1])
        o['risk_at_common_coverage'] = float(err[k - 1])
    if curve: o['curve'] = [[float(cov[i]), float(err[i])] for i in np.unique(np.linspace(0, n_ - 1, 120).astype(int))]
    return o
METHODS = ['ours_token', 'earliest_plausible', 'earliest_plausible_iota', 'dialogue_only', 'dialogue_only_iota',
           'naive_overlap', 'naive_overlap_iota', 'direction_blind', 'paste_log']
RCS = {}
out['B29_standard_selective'] = {}
for name in TEST:
    selective(SETS[name]); rc = G_['_RC']; RCS[name] = rc
    c_common = min(rc_standard(rc[k])['coverage'] for k in METHODS)
    out['B29_standard_selective'][name] = {'common_coverage': c_common, **{k: rc_standard(rc[k], c_common, curve=name in ('base', 'ext')) for k in METHODS}}
    log('B29', name, round(c_common, 3), {k: round(v['risk_at_common_coverage'], 4) for k, v in out['B29_standard_selective'][name].items() if k != 'common_coverage'})


# ---------- B-30 attribution against earliest plausible attribution with abstention, standard form
def student_slices(STUD):
    L = [len(tokens(st['final'])) for st in STUD]; e = np.cumsum(L); return list(zip(np.r_[0, e[:-1]], e))
def aurc_common(rows, c): return rc_standard(rows, c)['aurc_common_range']
out['B30_rules_vs_epa_abstention'] = {}
rb = np.random.default_rng(2174)
for name in ('base', 'ext'):
    rc = RCS[name]; sl = student_slices(SETS[name]); routes = [r for st in SETS[name] for r in st['route']]
    O, E = rc['ours_token'], rc['earliest_plausible_iota']; c = min(rc_standard(O)['coverage'], rc_standard(E)['coverage'])
    def diff(keep_idx):
        o_ = [O[i] for i in keep_idx]; e_ = [E[i] for i in keep_idx]; cc = min(rc_standard(o_)['coverage'], rc_standard(e_)['coverage'])
        return aurc_common(o_, cc) - aurc_common(e_, cc)
    res = {}
    for lab_, drop in (('all_routes', set()), ('without_shared_source_route', {'S'})):
        bs = []
        keepS = [np.array([i for i in range(a, b) if routes[i] not in drop]) for a, b in sl]
        full = diff(np.concatenate(keepS))
        for _ in range(NB30):
            idx = rb.integers(0, len(sl), len(sl)); bs.append(diff(np.concatenate([keepS[i] for i in idx])))
        res[lab_] = [float(full), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res['by_route_left_out'] = {rt: float(diff(np.array([i for i, r in enumerate(routes) if r != rt]))) for rt in sorted(set(routes))}
    out['B30_rules_vs_epa_abstention'][name] = res
    log('B30', name, res['all_routes'], res['without_shared_source_route'])


# ---------- B-31 the A3 set: false chatbot labels and the composition of the decided set
rc = RCS['a3']; routes = [r for st in SETS['a3'] for r in st['route']]
A3R = {'A3-after', 'A3-before', 'A3-regen'}
b31 = {}
for k in ('ours_token', 'earliest_plausible', 'earliest_plausible_iota'):
    rows = rc[k]; fc = sum(1 for (a, b, _) in rows if b and not a); fc_a3 = sum(1 for (a, b, _), r in zip(rows, routes) if b and not a and r in A3R)
    fn = sum(1 for (a, b, _) in rows if a and not b)
    c = out['B29_standard_selective']['a3']['common_coverage']
    order = sorted([(x, r) for x, r in zip(rows, routes) if x[2] != -1], key=lambda z: -z[0][2])[:int(c * len(rows))]
    b31[k] = {'false_chatbot_labels': fc, 'false_chatbot_labels_on_A3_routes': fc_a3, 'missed_chatbot_text': fn,
              'decided_at_common_coverage': len(order),
              'decided_correct_not_chatbot_share': float(np.mean([(not x[0]) and (not x[1]) for x, _ in order])),
              'decided_wrong_share': float(np.mean([x[0] != x[1] for x, _ in order]))}
out['B31_a3_decomposition'] = b31; log('B31', b31)


# ---------- evidence and the fusion classifier (as R16), with a hyperparameter search on the validation set
OWN = {}
for u, reps in REPL.items():
    for rep in reps:
        for g in set(ngrams(rep, N)): OWN.setdefault(g, set()).add(u)
ENC = F.Encoder(); POOL_TXT, POOL_U = [], []
for u, reps in REPL.items():
    for rep in reps:
        for a, b in F.windows(rep): POOL_TXT.append(' '.join(rep[a:b])); POOL_U.append(u)
POOL_U = np.array(POOL_U); POOL_E = ENC(POOL_TXT); ENC.save(); log('cohort pool', len(POOL_TXT))
class Fuzzy(dict):
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
def evidence(st):
    A = tokens(st['final']); o = attribute(st, st['snaps'], entry_rule='token', keep=True)
    fz = [c == 'C' for c in attribute_fuzzy(st)['cls']]
    s_any, s_dir = F.semantic(A, st['replies'], st['turns'], tokens, ENC)
    s_coh = F.cohort_max(A, ENC, POOL_E, POOL_U == st['donor'])
    cg = lambda g, u=st['donor']: bool(OWN.get(g, set()) - {u})
    X, aux = F.features(st, o, N, TAU, tokens, ngrams, fz, s_any, s_dir, cg, s_coh)
    y = np.array([F.CLASSES.index(c) for c in st['truth']])
    return dict(X=X, y=y, cls=o['cls'], route=st['route'], aux=aux)
EV = {name: [evidence(st) for st in STUD] for name, STUD in SETS.items()}; ENC.save(); log('evidence')
# B-36 subtypes of X without the two embedding flags: share of X and share truly C, on the test sets
b36 = {}
for name in TEST:
    tot, tc = Counter(), {}
    for e in EV[name]:
        for s_, y in zip(F.x_subtypes(e['cls'], e['aux'], 9.0, 9.0), e['y']):
            if s_ is None: continue
            tot[s_] += 1; tc.setdefault(s_, Counter())[F.CLASSES[y]] += 1
    nX = sum(tot.values()); b36[name] = {s_: {'share_of_X': tot[s_] / nX, 'truly_C': tc.get(s_, Counter())['C'] / max(1, tot[s_])} for s_ in F.XSUB}
out['B36_x_subtypes_without_embedding_flags'] = b36; log('B36', b36['base'])
def stack(ev): return np.vstack([e['X'] for e in ev]), np.concatenate([e['y'] for e in ev])
def f1_C(mdl, ev):
    Xs, ys = stack(ev); m = mdl.predict(Xs) == 0; t = ys == 0; tp = (m & t).sum()
    return float(tp / m.sum()), float(tp / t.sum()), float(2 * tp / (m.sum() + t.sum()))
from sklearn.ensemble import HistGradientBoostingClassifier
Xt, yt = stack(EV['train_base']); grid = []
for lr in (.05, .1, .2):
    for leaves in (15, 31, 63):
        for l2 in (0.0,):
            for mi in (300, 600):
                mdl = HistGradientBoostingClassifier(max_iter=mi, learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2, random_state=2160).fit(Xt, yt)
                p, r, f = f1_C(mdl, EV['val']); grid.append({'learning_rate': lr, 'max_leaf_nodes': leaves, 'l2': l2, 'max_iter': mi, 'val_precision': p, 'val_recall': r, 'val_f1': f})
    log('grid lr', lr)
best = max(grid, key=lambda g: g['val_f1']); R16DEF = [g for g in grid if g['learning_rate'] == .1 and g['max_leaf_nodes'] == 31 and g['l2'] == 0 and g['max_iter'] == 300][0]
TUNED = {}
for tr in ('train_base', 'train_all'):
    Xt_, yt_ = stack(EV[tr])
    TUNED[tr] = HistGradientBoostingClassifier(max_iter=best['max_iter'], learning_rate=best['learning_rate'], max_leaf_nodes=best['max_leaf_nodes'],
                                               l2_regularization=best['l2'], random_state=2160).fit(Xt_, yt_)
pickle.dump({'gbt_tuned_train_base': TUNED['train_base'], 'gbt_tuned_train_all': TUNED['train_all'], 'features': F.FEATURES, 'hyperparameters': best},
            open('results/r17_fusion_tuned.pkl', 'wb'))
R16M = pickle.load(open('results/r16_fusion_models.pkl', 'rb'))
b32 = {'grid': grid, 'best': best, 'r16_setting_on_validation': R16DEF, 'test': {}}
for name in TEST:
    b32['test'][name] = {'tuned_train_base': f1_C(TUNED['train_base'], EV[name]), 'tuned_train_all': f1_C(TUNED['train_all'], EV[name]),
                         'r16_train_base': f1_C(R16M['gbt_train_base'], EV[name]), 'r16_train_all': f1_C(R16M['gbt_train_all'], EV[name])}
    Xs, _ = stack(EV[name]); routes = [r for e in EV[name] for r in e['route']]
    if name == 'hard_x':
        for mk, mdl in (('tuned_train_base', TUNED['train_base']), ('r16_train_base', R16M['gbt_train_base'])):
            m = mdl.predict(Xs) == 0; b32['test'][name][mk + '_marks_on_X_mixed'] = float(np.mean([x for x, r in zip(m, routes) if r == 'X-mixed']))
out['B32_fusion_hyperparameters'] = b32; log('B32', best, b32['test']['base'])


# ---------- B-33 simulator with other outside text from a different source
xr = pd.read_csv(XNOEV); XN = {}
for u, s in zip(xr.user_id, xr.text): XN.setdefault(int(u), []).append(tokens(s))
ab = json.load(open(ARXIV)); AX = {f'arxiv{i}': [tokens(s)] for i, s in enumerate(ab) if len(tokens(s)) >= 40}
def rules_eval(STUD):
    cn, Ut, Ue, bc = [], [], [], []
    for st in STUD:
        cls = attribute(st, st['snaps'], entry_rule='token')['cls']; cn.append(counts(st['truth'], cls))
        Ut.append(st['truth'].count('C') / len(cls)); Ue.append(cls.count('C') / len(cls))
    o = {c: dict(zip(('precision', 'recall', 'f1'), prf_from(cn, c))) for c in ORDER}; o['U_mae'] = student_err(Ut, Ue)['mae']; return o
b33 = {'sources': {'original': 'unresolved runs of other FLoRA proposals (R8 labels)', 'x_no_evidence': f'{sum(len(v) for v in XN.values())} X-no-evidence runs of {len(XN)} FLoRA students',
                   'arxiv': f'{len(AX)} arXiv abstracts (stat.*, cs.LG/AI/IR/CY/DB/HC/CL/SI)'}, 'results': {}}
for src_name, pool in (('original', None), ('x_no_evidence', XN), ('arxiv', AX)):
    S2 = SETS if pool is None else build(TEST, pool)
    r = {}
    for name in TEST:
        r[name] = {'rules': rules_eval(S2[name])}
        selective(S2[name]); rc2 = G_['_RC']; c = min(rc_standard(rc2[k])['coverage'] for k in ('ours_token', 'earliest_plausible_iota'))
        r[name]['standard_selective'] = {k: rc_standard(rc2[k], c) for k in ('ours_token', 'earliest_plausible_iota', 'earliest_plausible')}
        r[name]['standard_selective']['common_coverage'] = c
        if name in ('base', 'hard_x'):
            ev2 = EV[name] if pool is None else [evidence(st) for st in S2[name]]
            r[name]['fusion_r16_train_base'] = f1_C(R16M['gbt_train_base'], ev2); r[name]['fusion_r16_train_all'] = f1_C(R16M['gbt_train_all'], ev2)
            if name == 'hard_x':
                Xs, _ = stack(ev2); routes = [q for e in ev2 for q in e['route']]; m = R16M['gbt_train_base'].predict(Xs) == 0
                r[name]['fusion_r16_train_base_marks_on_X_mixed'] = float(np.mean([x for x, q in zip(m, routes) if q == 'X-mixed']))
        log('B33', src_name, name, {c_: round(v['precision'], 3) for c_, v in r[name]['rules'].items() if c_ in ('C', 'X', 'T')})
    b33['results'][src_name] = r
ENC.save()
out['B33_outside_text_source'] = b33

out['seconds'] = time.perf_counter() - t00
json.dump(out, open('results/r17_benchmark_quick.json' if Q else 'results/r17_benchmark.json', 'w'), indent=1, default=float)
log('done')
