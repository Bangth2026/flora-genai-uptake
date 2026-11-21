"""Analysis log R18, B-37 and B-38 (benchmark).
B-37 the paired difference in the area under the standard risk--coverage curve, attribution minus earliest plausible
attribution with abstention by entry size (R17 B-30), with 2,000 paired student resamples instead of 300;
B-38 student-bootstrap intervals (2,000 resamples) for chatbot-text precision, recall and F1 of the evidence-fusion
classifiers trained on 13 and on all 26 routes (R16 default and R17 tuned settings, not retrained), and for their paired
F1 difference with the attribution, on the sets with 13 and with 20 routes.
The simulated students of seeds 2100 and 2200 are rebuilt exactly as in R16 and R17; nothing in the attribution changes.
Usage: EMB_CACHE=FILE python -I src/r18_benchmark.py DATA_DIR R8_TOKEN_CLASSES WORK_DIR -> results/r18_benchmark.json"""
import sys, os, json, time, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DATA_DIR, R8, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
s17 = open(os.path.join(HERE, 'r17_benchmark.py')).read()
head = s17[:s17.index('# ---------- B-31')]
ev_src = s17[s17.index('# ---------- evidence and the fusion classifier'):s17.index("EV = {name:")]
head = head.replace("XNOEV, ARXIV, WORK = sys.argv[3], sys.argv[4], sys.argv[5]", "XNOEV = ARXIV = None; WORK = sys.argv[5]")
head = head.replace("NB30 = 20 if Q else 300", "NB30 = 20 if Q else 2000")
head = head.replace("SETS = build([s[0] for s in SPEC]); TEST = ['base', 'ext', 'hard_x', 'a3']", "SETS = build(['base', 'ext']); TEST = ['base', 'ext']")
head = head.replace("pickle.dump({k: SETS[k][:20] for k in ('base', 'ext', 'a3')}, open(f'{WORK}/r17_llm_students.pkl', 'wb'))", "")
head = head.replace("rb = np.random.default_rng(2174)", "rb = np.random.default_rng(2184)")
sys.argv = [os.path.join(HERE, 'r17_benchmark.py'), DATA_DIR, R8, '', '', WORK]
G = {'__name__': 'r17_part', '__file__': os.path.join(HERE, 'r17_benchmark.py')}
exec(compile(head, 'r17_benchmark.py', 'exec'), G)
exec(compile(ev_src, 'r17_benchmark.py', 'exec'), G)
log, SETS, F, attribute, tokens = G['log'], G['SETS'], G['F'], G['attribute'], G['tokens']
out = {'n_per_set': G['NS'], 'B37_rules_vs_epa_abstention_2000': G['out']['B30_rules_vs_epa_abstention'],
       'B29_check': {k: {m: G['out']['B29_standard_selective'][k][m].get('risk_at_common_coverage') for m in ('ours_token', 'earliest_plausible_iota', 'earliest_plausible')}
                     for k in ('base', 'ext')}}
log('B37', out['B37_rules_vs_epa_abstention_2000'])

EV = {name: [G['evidence'](st) for st in STUD] for name, STUD in SETS.items()}; G['ENC'].save(); log('evidence')
R16M = pickle.load(open(os.path.join(HERE, '..', 'results', 'r16_fusion_models.pkl'), 'rb'))
R17M = pickle.load(open(os.path.join(HERE, '..', 'results', 'r17_fusion_tuned.pkl'), 'rb'))
MODELS = {'r16_train_base': R16M['gbt_train_base'], 'r16_train_all': R16M['gbt_train_all'],
          'tuned_train_base': R17M['gbt_tuned_train_base'], 'tuned_train_all': R17M['gbt_tuned_train_all']}
rng = np.random.default_rng(2185); NB = 2000
def counts(pred, y):
    t = y == 0; return np.array([(pred & t).sum(), pred.sum(), t.sum()], float)
def f1(c): return 2 * c[0] / (c[1] + c[2]) if c[1] + c[2] else 0.0
b38 = {}
for name in ('base', 'ext'):
    ev = EV[name]; ys = [e['y'] for e in ev]
    att = np.array([counts(np.array([c == 'C' for c in e['cls']]), e['y']) for e in ev])
    res = {'attribution_f1': f1(att.sum(0))}
    for mk, mdl in MODELS.items():
        mc = np.array([counts(mdl.predict(e['X']) == 0, e['y']) for e in ev])
        bs_f, bs_d, bs_p, bs_r = [], [], [], []
        for _ in range(NB):
            i = rng.integers(0, len(ev), len(ev)); m_, a_ = mc[i].sum(0), att[i].sum(0)
            bs_f.append(f1(m_)); bs_d.append(f1(a_) - f1(m_)); bs_p.append(m_[0] / m_[1]); bs_r.append(m_[0] / m_[2])
        m_, a_ = mc.sum(0), att.sum(0); ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res[mk] = {'precision': [float(m_[0] / m_[1])] + ci(bs_p), 'recall': [float(m_[0] / m_[2])] + ci(bs_r),
                   'f1': [f1(m_)] + ci(bs_f), 'delta_f1_attribution_minus_model': [f1(a_) - f1(m_)] + ci(bs_d)}
    b38[name] = res; log('B38', name, {k: (round(v['f1'][0], 3), [round(x, 3) for x in v['delta_f1_attribution_minus_model']]) for k, v in res.items() if k != 'attribution_f1'})
out['B38_classifier_intervals'] = b38
out['seconds'] = time.perf_counter() - G['t00']
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r18_benchmark.json'), 'w'), indent=1, default=float)
log('done')
