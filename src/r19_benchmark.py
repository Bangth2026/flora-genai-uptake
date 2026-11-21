"""Analysis log R19, B-39 and B-40 (benchmark).
B-39 writer text that the attribution never labelled: the four test sets of R15 (13 routes, 20 routes, outside text
retyped or chunked or mixed, A3) rebuilt with the same seeds, but every route that draws the writer's own text (typed
text, echo, interleaving, undo) draws it from arXiv abstracts (src/r19_writer_corpus.py) instead of the typed runs of
other FLoRA proposals. Per-class precision, recall and F1 of the rules, student-level error of U, and standard-form
selective risk at a common coverage for the attribution and for earliest plausible attribution with and without abstention.
B-40 decision cost on all eight sets: for the decision "chatbot text", per method the false chatbot labels (FP) and the
missed chatbot tokens (FN); expected cost per 1,000 tokens when a false chatbot label costs r times a miss
(r = 1, 2, 5, 10, 20, 50); the ratio r* above which the attribution costs less than each method that makes more false chatbot labels, with a
student-bootstrap interval (2,000 resamples; not defined, null, for a method with no more false chatbot labels than the
attribution, such as the audit operating point). Methods: the attribution (full attribution and audit operating point),
earliest plausible attribution, naive overlap, the paste-log rule, tolerant matching, and the evidence-fusion classifiers
tuned in R17 (trained on 13 and on 26 routes; not retrained). Abstention counts as "not chatbot".
Nothing in the attribution changes. Usage: EMB_CACHE=FILE python -I src/r19_benchmark.py DATA_DIR R8_TOKEN_CLASSES WRITER_JSON WORK_DIR
-> results/r19_benchmark.json"""
import sys, os, json, time, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DATA_DIR, R8, WRITER, WORK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s17 = open(os.path.join(HERE, 'r17_benchmark.py')).read()
head = s17[:s17.index("out['B30_rules_vs_epa_abstention'] = {}")]
ev_src = s17[s17.index('# ---------- evidence and the fusion classifier'):s17.index("EV = {name:")]
head = head.replace("XNOEV, ARXIV, WORK = sys.argv[3], sys.argv[4], sys.argv[5]", "XNOEV = ARXIV = None; WORK = sys.argv[5]")
head = head.replace("SETS = build([s[0] for s in SPEC]); TEST = ['base', 'ext', 'hard_x', 'a3']", "SETS = build(['base', 'ext', 'hard_x', 'a3']); TEST = ['base', 'ext', 'hard_x', 'a3']")
sys.argv = [os.path.join(HERE, 'r17_benchmark.py'), DATA_DIR, R8, '', '', WORK]
G = {'__name__': 'r17_part', '__file__': os.path.join(HERE, 'r17_benchmark.py')}
exec(compile(head, 'r17_benchmark.py', 'exec'), G)
log, SETS, G_, tokens, ngrams, N = G['log'], G['SETS'], G['G_'], G['tokens'], G['ngrams'], G['N']
out = {'n_per_set': G['NS'], 'B29_check_original': {k: {m: v[m]['risk_at_common_coverage'] for m in ('ours_token', 'earliest_plausible_iota', 'earliest_plausible')}
                                                    for k, v in G['out']['B29_standard_selective'].items()}}
RCS = dict(G['RCS'])

# ---------- B-39 writer text from an independent corpus
ab = json.load(open(WRITER)); WP = {f'w{i}': [tokens(r['abstract'])] for i, r in enumerate(ab)}
orig_trun = G_['TRUN']; G_['TRUN'] = WP
try: SW = G['build'](G['TEST'])
finally: G_['TRUN'] = orig_trun
counts, prf_from, ORDER, student_err, attribute = G_['counts'], G_['prf_from'], G_['ORDER'], G_['student_err'], G_['attribute']
def rules_eval(STUD):
    cn, Ut, Ue = [], [], []
    for st in STUD:
        cls = attribute(st, st['snaps'], entry_rule='token')['cls']; cn.append(counts(st['truth'], cls))
        Ut.append(st['truth'].count('C') / len(cls)); Ue.append(cls.count('C') / len(cls))
    o = {c: dict(zip(('precision', 'recall', 'f1'), prf_from(cn, c))) for c in ORDER}; o['U_error'] = student_err(Ut, Ue); return o
b39 = {'writer_corpus': f'{len(WP)} arXiv abstracts (gfissore/arxiv-abstracts-2021, seed 2193)'}
for name in G['TEST']:
    r = {'rules_original': rules_eval(SETS[name]), 'rules_independent_writer_text': rules_eval(SW[name])}
    G_['selective'](SW[name]); rc = G_['_RC']; RCS['w_' + name] = rc
    c = min(G['rc_standard'](rc[k])['coverage'] for k in ('ours_token', 'earliest_plausible_iota'))
    r['standard_selective_independent_writer_text'] = {k: G['rc_standard'](rc[k], c) for k in ('ours_token', 'earliest_plausible_iota', 'earliest_plausible')}
    r['standard_selective_independent_writer_text']['common_coverage'] = c
    b39[name] = r
    log('B39', name, {cc: (round(r['rules_original'][cc]['precision'], 3), round(r['rules_independent_writer_text'][cc]['precision'], 3), round(r['rules_independent_writer_text'][cc]['recall'], 3)) for cc in 'CETX'})
out['B39_independent_writer_text'] = b39

# ---------- B-40 decision cost
exec(compile(ev_src, 'r17_benchmark.py', 'exec'), G)
R17M = pickle.load(open(os.path.join(HERE, '..', 'results', 'r17_fusion_tuned.pkl'), 'rb'))
MODELS = {'classifier_13_routes': R17M['gbt_tuned_train_base'], 'classifier_26_routes': R17M['gbt_tuned_train_all']}
ALLSETS = {**SETS, **{'w_' + k: v for k, v in SW.items()}}
RULE_METHODS = {'attribution': 'ours_token', 'earliest_plausible': 'earliest_plausible', 'naive_overlap': 'naive_overlap', 'paste_log': 'paste_log'}
def cover_count(L, i): return max(0, min(i + 1, N, L - i, L - N + 1))
CNT = {}
for name, STUD in ALLSETS.items():
    rc = RCS[name]; sl = G['student_slices'](STUD); per = {m: [] for m in list(RULE_METHODS) + ['audit', 'tolerant'] + list(MODELS)}
    for (a, b), st in zip(sl, STUD):
        L = b - a
        for m, key in RULE_METHODS.items():
            rows = rc[key][a:b]; y = np.array([x[0] for x in rows]); p = np.array([x[1] for x in rows])
            per[m].append((int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), L))
        rows = rc['ours_token'][a:b]; y = np.array([x[0] for x in rows])
        p = np.array([x[1] and x[2] == cover_count(L, i) for i, x in enumerate(rows)]); per['audit'].append((int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), L))
        p = np.array([c == 'C' for c in G['attribute_fuzzy'](st)['cls']]); per['tolerant'].append((int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), L))
        e = G['evidence'](st)
        for mk, mdl in MODELS.items():
            p = mdl.predict(e['X']) == 0; per[mk].append((int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), L))
    CNT[name] = {m: np.array(v) for m, v in per.items()}; G['ENC'].save(); log('B40 counts', name)
RS = (1, 2, 5, 10, 20, 50); rng = np.random.default_rng(2194); NB = 2000
def cost(c, r): t = c.sum(0); return 1000 * (r * t[1] + t[2]) / t[3]
def rstar(cR, cM):
    tR, tM = cR.sum(0), cM.sum(0); dfp = tM[1] - tR[1]; dfn = tR[2] - tM[2]
    if dfp > 0: return float(max(dfn / dfp, 0.0))
    return float('nan')   # the method makes no more false chatbot labels than the attribution: r* is not defined
b40 = {}
for name, C in CNT.items():
    o = {m: {'precision': float(c.sum(0)[0] / max(c.sum(0)[0] + c.sum(0)[1], 1)), 'recall': float(c.sum(0)[0] / max(c.sum(0)[0] + c.sum(0)[2], 1)),
             'cost_per_1000_tokens': {str(r): cost(c, r) for r in RS}} for m, c in C.items()}
    o['ratio_above_which_attribution_costs_less'] = {}
    for m in C:
        if m == 'attribution': continue
        pt = rstar(C['attribution'], C[m]); bs = []
        for _ in range(NB):
            i = rng.integers(0, len(C[m]), len(C[m])); bs.append(rstar(C['attribution'][i], C[m][i]))
        o['ratio_above_which_attribution_costs_less'][m] = [pt, float(np.percentile(bs, 2.5, method='inverted_cdf')), float(np.percentile(bs, 97.5, method='inverted_cdf'))]
    o['lowest_cost_method'] = {str(r): min(C, key=lambda m: cost(C[m], r)) for r in RS}
    b40[name] = o; log('B40', name, o['lowest_cost_method'], {k: [round(x, 2) for x in v] for k, v in o['ratio_above_which_attribution_costs_less'].items() if k.startswith('class')})
out['B40_decision_cost'] = b40
out['seconds'] = time.perf_counter() - G['t00']
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r19_benchmark.json'), 'w'), indent=1, default=float)
log('done')
