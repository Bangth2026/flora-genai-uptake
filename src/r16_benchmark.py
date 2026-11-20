"""Analysis log R16, B-22 to B-28 (benchmark).
B-22 subtypes of the unresolved class by route; B-23 operating points (audit, screening, full attribution); B-24
one-at-a-time and joint sensitivity of n, tau and delta; B-25 tolerant and embedding baselines with thresholds tuned on a
separate validation set and student-bootstrap intervals; B-26 an evidence-fusion classifier (gradient-boosted trees over
the evidence features of src/r16_features.py) trained on separate simulated students and tested on the four R15 sets,
including routes it never saw; B-27 calibration; B-28 permutation importance of the fusion features.

The students of seeds 2100, 2200, 2300 and 2400 are rebuilt exactly as in R15 (same generator, same route branches).
New seeds: 2500 (13 specified routes, training), 2600 (all 26 routes, training), 2700 (13 routes, validation for
thresholds). Written after R15 and before these analyses ran; see ANALYSIS_LOG_R16.md.
Usage: python -I src/r16_benchmark.py DATA_DIR R8_TOKEN_CLASSES -> results/r16_benchmark.json"""
import sys, os, json, random, time, pickle
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r16_features as F

# ---------- generator and attribution of R11 with the route branches of R15 (copied from src/r15_benchmark.py)
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
for k in ('make_student', 'BASE', 'EXT', 'attribute', 'scan', 'first_times', 'tokens', 'ngrams', 'N', 'TAU', 'ORDER',
          'counts', 'prf_from', 'bin_counts', 'bin_prf', 'student_err', 'REPL', 'SHARED_TXT'):
    globals()[k] = G_[k]
Q = int(os.environ.get('QUICK', 0)); NS = 40 if Q else 400; NB = 200 if Q else 2000
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)

HARDX = BASE + [('X-retype', .05), ('X-chunks', .05), ('X-mixed', .05)]
A3 = BASE + [('A3-after', .05), ('A3-before', .05), ('A3-regen', .05)]
ALL = EXT + HARDX[-3:] + A3[-3:]
SETS = {}
for name, seed, routes in [('base', 2100, BASE), ('ext', 2200, EXT), ('hard_x', 2300, HARDX), ('a3', 2400, A3),
                           ('train_base', 2500, BASE), ('train_all', 2600, ALL), ('val', 2700, BASE)]:
    R = random.Random(seed); SETS[name] = [make_student(R, routes) for _ in range(NS)]
TEST = ['base', 'ext', 'hard_x', 'a3']
log('students built')
out = {'n_per_set': NS, 'seeds': {'base': 2100, 'ext': 2200, 'hard_x': 2300, 'a3': 2400, 'train_base': 2500, 'train_all': 2600, 'val': 2700}}


# ---------- tolerant matching (as R15 B-20)
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


# ---------- cohort: replies given to other FLoRA writers (the pool from which the outside-AI route draws)
OWN = {}
for u, reps in REPL.items():
    for rep in reps:
        for g in set(ngrams(rep, N)): OWN.setdefault(g, set()).add(u)
ENC = F.Encoder()
POOL_TXT, POOL_U = [], []
for u, reps in REPL.items():
    for rep in reps:
        for a, b in F.windows(rep): POOL_TXT.append(' '.join(rep[a:b])); POOL_U.append(u)
POOL_U = np.array(POOL_U); POOL_E = ENC(POOL_TXT); ENC.save()
log('cohort pool', len(POOL_TXT))


def evidence(st):
    A = tokens(st['final']); o = attribute(st, st['snaps'], entry_rule='token', keep=True)
    fz = [c == 'C' for c in attribute_fuzzy(st)['cls']]
    s_any, s_dir = F.semantic(A, st['replies'], st['turns'], tokens, ENC)
    s_coh = F.cohort_max(A, ENC, POOL_E, POOL_U == st['donor'])
    cg = lambda g, u=st['donor']: bool(OWN.get(g, set()) - {u})
    X, aux = F.features(st, o, N, TAU, tokens, ngrams, fz, s_any, s_dir, cg, s_coh)
    y = np.array([F.CLASSES.index(c) for c in st['truth']])
    return dict(X=X, y=y, cls=o['cls'], agree=o['agree'], aux=aux, route=st['route'], fuzzy=fz, s_any=s_any, s_dir=s_dir)


EV = {}
for name, STUD in SETS.items():
    EV[name] = [evidence(st) for st in STUD]; log('evidence', name)


def boot_prf(per, rng, nb=NB):
    """per-student binary counts (tp, pred, actual) -> F1 and 95% interval over students"""
    per = np.array(per, dtype=float); vals = []
    for _ in range(nb):
        s = per[rng.integers(0, len(per), len(per))].sum(0); p = s[0] / s[1] if s[1] else np.nan; r = s[0] / s[2]
        vals.append(2 * p * r / (p + r) if p + r else np.nan)
    s = per.sum(0); p, r = s[0] / s[1], s[0] / s[2]
    return {'precision': float(p), 'recall': float(r), 'f1': float(2 * p * r / (p + r)),
            'f1_ci': [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}


def boot_diff(perA, perB, rng, nb=NB):
    a, b = np.array(perA, float), np.array(perB, float); vals = []
    f = lambda s: 2 * s[0] / (s[1] + s[2])
    for _ in range(nb):
        i = rng.integers(0, len(a), len(a)); vals.append(f(a[i].sum(0)) - f(b[i].sum(0)))
    return [float(f(a.sum(0)) - f(b.sum(0))), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------- B-22 subtypes of X by route
def tune_flag(ev, key, target):
    """threshold on the semantic score that maximises F1 of the flag for a target among X tokens"""
    sc, tg = [], []
    for e in ev:
        for j, c in enumerate(e['cls']):
            if c == 'X': sc.append(e['aux'][key][j]); tg.append(target(e, j))
    sc, tg = np.array(sc), np.array(tg); best = (0, None)
    for th in np.arange(.50, .99, .01):
        m = sc >= th; tp_ = (m & tg).sum(); p = tp_ / m.sum() if m.sum() else 0; r = tp_ / tg.sum() if tg.sum() else 0
        f = 2 * p * r / (p + r) if p + r else 0
        if f > best[0]: best = (f, float(round(th, 2)), float(p), float(r))
    return best
own_t = tune_flag(EV['val'], 's_dir', lambda e, j: e['route'][j] in ('C-para10', 'C-para25', 'C-paste', 'C-retype'))
coh_t = tune_flag(EV['val'], 's_cohort', lambda e, j: e['route'][j] == 'X-ai')
TH_OWN, TH_COH = own_t[1], coh_t[1]
out['B22_thresholds_validation'] = {'near_own_reply': dict(zip(('f1', 'theta', 'precision', 'recall'), own_t)),
                                    'cohort_reply': dict(zip(('f1', 'theta', 'precision', 'recall'), coh_t))}
log('thresholds', own_t, coh_t)
out['B22_x_subtypes'] = {}
for name in TEST:
    byr, tot, truth_by_sub = {}, Counter(), {}
    for e in EV[name]:
        sub = F.x_subtypes(e['cls'], e['aux'], TH_OWN, TH_COH)
        for r, s, y in zip(e['route'], sub, e['y']):
            if s is None: continue
            byr.setdefault(r, Counter())[s] += 1; tot[s] += 1; truth_by_sub.setdefault(s, Counter())[F.CLASSES[y]] += 1
    nX = sum(tot.values())
    out['B22_x_subtypes'][name] = {'share_of_X': {s: tot[s] / nX for s in F.XSUB},
                                   'true_class_within_subtype': {s: {c: v / sum(cc.values()) for c, v in cc.items()} for s, cc in truth_by_sub.items()},
                                   'route_to_subtype': {r: {s: v / sum(cc.values()) for s, v in cc.items()} for r, cc in byr.items()}}
log('B22 done')


# ---------- B-23 operating points, decision "chatbot text"
def op_points(ev):
    res = {}
    for mode in ('audit', 'screening', 'full'):
        dec = err = tp_ = pp = ap = 0; N_ = 0
        for e in ev:
            for c, ag, y in zip(e['cls'], e['agree'], e['y']):
                N_ += 1; isC = y == 0; ap += isC
                if mode == 'audit': d = (c == 'C' and ag >= N) or c == 'E'; mark = c == 'C' and ag >= N
                elif mode == 'screening': d = c != 'X'; mark = c == 'C'
                else: d = True; mark = c == 'C'
                if d: dec += 1; err += (mark != isC)
                tp_ += mark and isC; pp += mark
        res[mode] = {'coverage': dec / N_, 'selective_risk': err / dec, 'precision_C': tp_ / pp, 'recall_C': tp_ / ap}
    return res
out['B23_operating_points'] = {name: op_points(EV[name]) for name in TEST}
log('B23 done', out['B23_operating_points']['base'])


# ---------- B-25 baselines with thresholds tuned on validation, intervals over students
rng = np.random.default_rng(2160)
def per_student(ev, markfn): return [bin_counts(list(e['y'] == 0), list(markfn(e))) for e in ev]
grid = np.round(np.arange(.60, .96, .05), 2); tuned = {}
for kind in ('s_any', 's_dir'):
    f1s = {float(th): bin_prf(per_student(EV['val'], lambda e: e[kind] >= th))[2] for th in grid}
    tuned[kind] = {'theta': max(f1s, key=f1s.get), 'validation_f1_by_theta': f1s}
out['B25_tuning'] = tuned
out['B25_baselines'] = {}
for name in TEST:
    ours = per_student(EV[name], lambda e: np.array(e['cls']) == 'C'); r = {'attribution': boot_prf(ours, rng)}
    for lab_, fn in [('tolerant', lambda e: np.array(e['fuzzy'])),
                     ('retrieval_any_tuned', lambda e: e['s_any'] >= tuned['s_any']['theta']),
                     ('retrieval_dir_tuned', lambda e: e['s_dir'] >= tuned['s_dir']['theta'])]:
        ps = per_student(EV[name], fn); r[lab_] = boot_prf(ps, rng); r[lab_]['delta_f1_attribution_minus'] = boot_diff(ours, ps, rng)
    out['B25_baselines'][name] = r
log('B25 done', {k: round(v['f1'], 3) for k, v in out['B25_baselines']['base'].items()})


# ---------- B-26/27/28 evidence-fusion classifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def stack(ev): return np.vstack([e['X'] for e in ev]), np.concatenate([e['y'] for e in ev])
MODELS = {}
for tr in ('train_base', 'train_all'):
    Xt, yt = stack(EV[tr])
    MODELS[f'gbt_{tr}'] = HistGradientBoostingClassifier(max_iter=300, learning_rate=.1, max_leaf_nodes=31, random_state=2160).fit(Xt, yt)
    MODELS[f'logit_{tr}'] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Xt, yt)
    log('trained', tr, len(yt))
pickle.dump({'gbt_train_base': MODELS['gbt_train_base'], 'gbt_train_all': MODELS['gbt_train_all'], 'features': F.FEATURES,
             'theta_own': TH_OWN, 'theta_cohort': TH_COH}, open('results/r16_fusion_models.pkl', 'wb'))


def selective(conf, mark, truthC, c_common):
    o = np.argsort(-conf, kind='stable'); e = (mark[o] != truthC[o]).astype(float)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1); cov = np.arange(1, len(e) + 1) / len(e)
    k = max(1, int(np.searchsorted(cov, c_common, side='right')))
    return {'risk_at_common_coverage': float(risk[k - 1]), 'aurc_common_range': float(np.trapezoid(risk[:k], cov[:k]) / cov[k - 1]),
            'aurc_full': float(np.trapezoid(risk, cov)), 'error_all_decided': float(risk[-1])}


out['B26_fusion'] = {}; out['B27_calibration'] = {}
for name in TEST:
    Xs, ys = stack(EV[name]); res = {}; cal = {}
    rule = np.concatenate([np.array(e['cls']) for e in EV[name]]); agree = np.concatenate([np.array(e['agree']) for e in EV[name]])
    truthC = ys == 0; ruleC = rule == 'C'
    # rule base as selective classifier, as in R15: abstain on X, rank by agreeing covering n-grams
    rconf = np.where(rule == 'X', -1, agree).astype(float); c_common = float((rule != 'X').mean())
    res['attribution'] = {'per_class': {c: dict(zip(('precision', 'recall', 'f1'), prf_from([counts([F.CLASSES[y] for y in ys], list(rule))], c))) for c in F.CLASSES},
                          'selective': selective(rconf, ruleC, truthC, c_common), 'coverage_rule': c_common}
    # reliability of the rule base: precision of C by number of agreeing covering n-grams
    cal['attribution_C_precision_by_agree'] = {int(a): [int((ruleC & (agree == a)).sum()), float(truthC[ruleC & (agree == a)].mean())] for a in range(1, N + 1) if (ruleC & (agree == a)).any()}
    for mk, mdl in MODELS.items():
        P = mdl.predict_proba(Xs); pred = P.argmax(1); mark = pred == 0
        per = []; k0 = 0
        for e in EV[name]:
            k1 = k0 + len(e['y']); per.append(bin_counts(list(e['y'] == 0), list(mark[k0:k1]))); k0 = k1
        b = boot_prf(per, rng); b['delta_f1_attribution_minus'] = boot_diff(per_student(EV[name], lambda e: np.array(e['cls']) == 'C'), per, rng)
        res[mk] = {'chatbot_text': b,
                   'per_class': {c: dict(zip(('precision', 'recall', 'f1'), prf_from([counts([F.CLASSES[y] for y in ys], [F.CLASSES[p] for p in pred])], c))) for c in F.CLASSES},
                   'accuracy': float((pred == ys).mean()),
                   'selective': selective(np.abs(P[:, 0] - .5), mark, truthC, c_common),
                   'student_U': student_err([float((e['y'] == 0).mean()) for e in EV[name]],
                                            [float(m_.mean()) for m_ in np.split(mark, np.cumsum([len(e['y']) for e in EV[name]])[:-1])])}
        e_top, rel_top = F.ece(P, ys); e_c, rel_c = F.ece_binary(P[:, 0], truthC.astype(float))
        cal[mk] = {'ece_top_label': e_top, 'ece_chatbot_prob': e_c, 'reliability_chatbot_prob': rel_c}
    res['attribution']['accuracy'] = float((rule == np.array([F.CLASSES[y] for y in ys])).mean())
    out['B26_fusion'][name] = res; out['B27_calibration'][name] = cal
    log('B26', name, {k: round(v['chatbot_text']['f1'], 3) for k, v in res.items() if k != 'attribution'}, 'rule', round(res['attribution']['per_class']['C']['f1'], 3))
# fusion by route on the extended and A3 sets (where routes are unseen by the model trained on the 13 routes)
out['B26_fusion_route_marked_C'] = {}
for name in ('ext', 'hard_x', 'a3'):
    Xs, ys = stack(EV[name]); routes = [r for e in EV[name] for r in e['route']]; rule = np.concatenate([np.array(e['cls']) for e in EV[name]])
    d = {}
    for mk in ('gbt_train_base', 'gbt_train_all'):
        mark = MODELS[mk].predict_proba(Xs).argmax(1) == 0; c = {}
        for r, m_ in zip(routes, mark): c.setdefault(r, [0, 0]); c[r][0] += m_; c[r][1] += 1
        d[mk] = {r: v[0] / v[1] for r, v in c.items()}
    c = {}
    for r, m_ in zip(routes, rule == 'C'): c.setdefault(r, [0, 0]); c[r][0] += m_; c[r][1] += 1
    d['attribution'] = {r: v[0] / v[1] for r, v in c.items()}
    out['B26_fusion_route_marked_C'][name] = d

# B-28 permutation importance (gbt trained on 13 routes, base test set, F1 of the chatbot-text decision)
Xs, ys = stack(EV['base']); idx = np.random.default_rng(2161).choice(len(ys), min(60000, len(ys)), replace=False)
Xb, yb = Xs[idx], ys[idx] == 0; mdl = MODELS['gbt_train_base']
def f1(m_, y_): tp_ = (m_ & y_).sum(); return 2 * tp_ / (m_.sum() + y_.sum())
base_f1 = f1(mdl.predict(Xb) == 0, yb); imp = {}
prng = np.random.default_rng(2162)
for j, fn in enumerate(F.FEATURES):
    Xp = Xb.copy(); Xp[:, j] = prng.permutation(Xp[:, j]); imp[fn] = float(base_f1 - f1(mdl.predict(Xp) == 0, yb))
out['B28_permutation_importance_f1_drop'] = dict(sorted(imp.items(), key=lambda x: -x[1]))
log('B28 done')


# ---------- B-24 sensitivity of n, tau, delta (rule base only)
def run_cfg(STUD, n, tau, delta):
    G_['SHG'] = {g for s in SHARED_TXT for g in ngrams(tokens(s), n)}
    cn, Ut, Ue = [], [], []
    for st in STUD:
        cls = attribute(st, st['snaps'], n=n, tau=tau, delta=delta, entry_rule='token')['cls']
        cn.append(counts(st['truth'], cls)); Ut.append(st['truth'].count('C') / len(cls)); Ue.append(cls.count('C') / len(cls))
    tot = sum(sum(x[c][2] for c in ORDER) for x in cn); corr = sum(sum(x[c][0] for c in ORDER) for x in cn)
    return {'accuracy': corr / tot, **{f'{c}_{m}': v for c in ORDER for m, v in zip(('p', 'r', 'f1'), prf_from(cn, c))},
            'U_mae': student_err(Ut, Ue)['mae']}
cfgs = sorted(set([(n, 5, 0.0) for n in (5, 6, 8, 10, 12)] + [(8, t, 0.0) for t in (3, 5, 10, 20)] +
                  [(8, 5, d) for d in (0.0, 5.0, 30.0)] + [(n, t, 0.0) for n in (5, 8, 12) for t in (3, 10, 20)]))
out['B24_sensitivity'] = {}
for name in ('base', 'ext'):
    out['B24_sensitivity'][name] = {f'n{n}_tau{t}_delta{int(d)}': run_cfg(SETS[name], n, t, d) for n, t, d in cfgs}
    log('B24', name)
G_['SHG'] = {g for s in SHARED_TXT for g in ngrams(tokens(s), N)}

out['seconds'] = time.perf_counter() - t00
json.dump(out, open('results/r16_benchmark_quick.json' if Q else 'results/r16_benchmark.json', 'w'), indent=1, default=float)
log('done')
