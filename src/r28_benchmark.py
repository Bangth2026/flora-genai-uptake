"""Analysis log R28, B-48 (benchmark), written after a review of the thirteenth version of the manuscript asked for it:
tolerant matching (one token of each 8-gram may differ; R15 B-20) with a confidence score, so that it can be compared
with the attribution on the risk--coverage curve. Tolerant matching runs the decision list with tolerant first-occurrence
times and abstains on X, as the attribution does; a decided token is ranked by the number of covering 8-grams that
carry its label under exact matching (0 when exact matching gives the token another label, so decisions that rest on
tolerant matches alone come last). The simulated students of seeds 2100 (13 routes) and 2200 (20 routes) are rebuilt
exactly as in R16 to R18; the common coverage c* is that of R17 B-29 (the smallest coverage among the methods with
abstention). Nothing in the attribution changes.
Usage: python -I src/r28_benchmark.py DATA_DIR R8_TOKEN_CLASSES WORK_DIR -> results/r28_benchmark.json
(R8_TOKEN_CLASSES: git show af6bdb9:results/flora_token_classes.csv.gz > FILE, as for R12-R22)"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DATA_DIR, R8, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
s17 = open(os.path.join(HERE, 'r17_benchmark.py')).read()
head = s17[:s17.index('# ---------- B-30')]
head = head.replace("XNOEV, ARXIV, WORK = sys.argv[3], sys.argv[4], sys.argv[5]", "XNOEV = ARXIV = None; WORK = sys.argv[5]")
head = head.replace("SETS = build([s[0] for s in SPEC]); TEST = ['base', 'ext', 'hard_x', 'a3']", "SETS = build(['base', 'ext']); TEST = ['base', 'ext']")
head = head.replace("pickle.dump({k: SETS[k][:20] for k in ('base', 'ext', 'a3')}, open(f'{WORK}/r17_llm_students.pkl', 'wb'))", "")
head = head.replace("import r16_features as F", "F = None")
fz = s17[s17.index('class Fuzzy(dict):'):s17.index('def evidence(st):')]
sys.argv = [os.path.join(HERE, 'r17_benchmark.py'), DATA_DIR, R8, '', '', WORK]
G = {'__name__': 'r17_part', '__file__': os.path.join(HERE, 'r17_benchmark.py')}
exec(compile(head, 'r17_benchmark.py', 'exec'), G)
exec(compile(fz, 'r17_benchmark.py', 'exec'), G)
log, SETS, attribute, attribute_fuzzy, rc_standard, RCS, tokens = (G[k] for k in ('log', 'SETS', 'attribute', 'attribute_fuzzy', 'rc_standard', 'RCS', 'tokens'))
out = {'n_per_set': G['NS'], 'B48_tolerant_selective': {}}
for name in ('base', 'ext'):
    rows, cnt = [], np.zeros(3)
    for st in SETS[name]:
        ex = attribute(st, st['snaps'], entry_rule='token', keep=True); tol = attribute_fuzzy(st)['cls']
        truthC = [x == 'C' for x in st['truth']]
        assert len(tol) == len(ex['cls']) == len(truthC)
        for a, ce, ag, ct in zip(truthC, ex['cls'], ex['agree'], tol):
            rows.append((a, ct == 'C', -1 if ct == 'X' else (ag if ce == ct else 0)))
            cnt += [a and ct == 'C', ct == 'C', a]
    b29 = G['out']['B29_standard_selective'][name]; c = b29['common_coverage']
    f1 = 2 * cnt[0] / (cnt[1] + cnt[2])
    out['B48_tolerant_selective'][name] = {'common_coverage': c, 'tolerant_f1_full': float(f1),
        'tolerant': rc_standard(rows, c), 'attribution': rc_standard(RCS[name]['ours_token'], c),
        'earliest_plausible_iota': rc_standard(RCS[name]['earliest_plausible_iota'], c),
        'tolerant_at_attribution_coverage': rc_standard(rows, rc_standard(RCS[name]['ours_token'])['coverage'])}
    log('B48', name, {k: (round(v.get('coverage', 0), 3), round(v.get('risk_at_common_coverage', -1), 4)) for k, v in out['B48_tolerant_selective'][name].items() if isinstance(v, dict)}, round(f1, 3))
out['seconds'] = time.perf_counter() - G['t00']
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r28_benchmark.json'), 'w'), indent=1, default=float)
log('done')
