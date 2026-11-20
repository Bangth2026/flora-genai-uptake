"""Analysis log R17, B-35 (comparison): the attribution and earliest plausible attribution on the same simulated
students as the LLM baseline (first NSTUD students of each set in WORK_DIR/r17_llm_students.pkl), with the same
token-level scoring and student-bootstrap intervals, and the paired difference in F1 against the LLM.
Usage: NSTUD=12 python -I src/r17_llm_compare.py DATA_DIR R8_TOKEN_CLASSES WORK_DIR -> results/r17_llm_compare.json"""
import sys, os, json, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'r11_benchmark.py')).read().replace("'results/flora_token_classes.csv.gz'", repr(sys.argv[2]))
cut = src.index("out = {'n_per_seed': 400}"); WORK = sys.argv[3]
sys.argv = [os.path.join(HERE, 'r11_benchmark.py'), sys.argv[1]]
G = {'__name__': 'r11_defs', '__file__': os.path.join(HERE, 'r11_benchmark.py')}
os.chdir(os.path.join(HERE, '..')); exec(compile(src[:cut], 'r11_benchmark.py', 'exec'), G)
NS = int(os.environ.get('NSTUD', 12)); SETS = {k: v[:NS] for k, v in pickle.load(open(f'{WORK}/r17_llm_students.pkl', 'rb')).items()}
L = json.load(open('results/r17_llm_baseline.json')); rng = np.random.default_rng(2176); out = {}
def f1(a): s = a.sum(0); return 2 * s[0] / (s[1] + s[2])
for name, STUD in SETS.items():
    per = []
    for st in STUD:
        cls = G['attribute'](st, st['snaps'], entry_rule='token')['cls']; t = [x == 'C' for x in st['truth']]; m = [c == 'C' for c in cls]
        per.append((sum(a and b for a, b in zip(t, m)), sum(m), sum(t)))
    a = np.array(per, float); s = a.sum(0); o = {'precision': s[0] / s[1], 'recall': s[0] / s[2], 'f1': f1(a)}
    if name in L['sets']:
        b = np.array(L['sets'][name]['per_student'], float); d = []
        for _ in range(2000):
            i = rng.integers(0, len(a), len(a)); d.append(f1(a[i]) - f1(b[i]))
        o['delta_f1_attribution_minus_llm'] = [float(f1(a) - f1(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    out[name] = {k: (float(v) if not isinstance(v, list) else v) for k, v in o.items()}; print(name, out[name], flush=True)
json.dump(out, open('results/r17_llm_compare.json', 'w'), indent=1)
