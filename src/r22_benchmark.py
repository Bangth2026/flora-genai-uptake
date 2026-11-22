"""Analysis log R22, B-47 (post hoc): tolerant matching on the set with chatbot text revised by an open language model
(R21 B-46). The set is rebuilt exactly as in src/r21_benchmark.py from the cached revisions (no model call), and the
decision "chatbot text" of the attribution is compared with the same decision list in which reply and prompt n-grams also
match with one token replaced (tolerant matching of R15, Supplementary Section B.3.2), on the whole set and per route.
Usage: NSTUD=100 python -I src/r22_benchmark.py DATA_DIR R8_TOKEN_CLASSES WORK_DIR -> results/r22_benchmark.json
(WORK_DIR/r21_llm_cache.json: gunzip results/r21_llm_revisions.json.gz)"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, R8, WORK = sys.argv[1:4]
src = open(os.path.join(HERE, 'r21_benchmark.py')).read()
head = src[:src.index("cn, Ut, Ue = [], [], []")]
sys.argv = [os.path.join(HERE, 'r21_benchmark.py'), DATA, R8, WORK, 'no-model-needed']
NS_ = {'__name__': 'r21', '__file__': os.path.join(HERE, 'r21_benchmark.py')}
exec(compile(head, 'r21_benchmark.py', 'exec'), NS_)
G_ = NS_['G_']; STUD = NS_['STUD']; attribute = NS_['attribute']; counts = NS_['counts']; prf_from = NS_['prf_from']

class Fuzzy(dict):
    """first time of each n-gram, matched exactly or with one token replaced (as src/r15_benchmark.py)"""
    def __init__(self, items, n):
        super().__init__()
        for t, text in sorted(items, key=lambda x: x[0]):
            for g in G_['ngrams'](G_['tokens'](text), n):
                for key in [g] + [g[:i] + ('\0',) + g[i + 1:] for i in range(n)]:
                    dict.setdefault(self, key, t)
    def _keys(self, g): return [g] + [g[:i] + ('\0',) + g[i + 1:] for i in range(len(g))]
    def __contains__(self, g): return any(dict.__contains__(self, k) for k in self._keys(g))
    def __getitem__(self, g): return min(dict.__getitem__(self, k) for k in self._keys(g) if dict.__contains__(self, k))

def run(tolerant):
    orig = G_['first_times']
    if tolerant: G_['first_times'] = lambda items, n: Fuzzy(items, n)
    try:
        cn, byr = [], {}
        for st in STUD:
            cls = attribute(st, st['snaps'], entry_rule='token')['cls']; cn.append(counts(st['truth'], cls))
            prev = None
            for r, c, t in zip(st['route'], cls, st['truth']):
                key = r if r != 'Tedit' else (prev + ':edit' if prev else 'Tedit')
                if r != 'Tedit': prev = r if r.startswith('C-llm') else None
                byr.setdefault(key, {}); byr[key][c] = byr[key].get(c, 0) + 1
        res = dict(zip(('precision', 'recall', 'f1'), prf_from(cn, 'C')))
        res['C_share_by_route'] = {r: d_.get('C', 0) / sum(d_.values()) for r, d_ in byr.items() if r.startswith('C-') or r.endswith(':edit') or r in ('T', 'X-other', 'X-ai', 'E-prompt', 'E-editor')}
        return res
    finally: G_['first_times'] = orig
out = {'n_students': len(STUD), 'rules': run(False), 'tolerant': run(True)}
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r22_benchmark.json'), 'w'), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'C_share_by_route'} if isinstance(v, dict) else v for k, v in out.items()}, indent=1))
for r in sorted(out['rules']['C_share_by_route']): print(r, round(out['rules']['C_share_by_route'][r], 3), round(out['tolerant']['C_share_by_route'].get(r, 0), 3))
