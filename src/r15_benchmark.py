"""Analysis log R15, B-17 to B-21 (benchmark): routes on which the definition of a class and the condition that
assigns it no longer coincide (outside text retyped, pasted in chunks of at most tau tokens, or pasted interleaved
with chatbot text); routes that violate assumption A3 (outside text identical to, or regenerated from, the embedded
chatbot's own reply); selective risk in the standard form (risk over decided tokens only, coverage below one); and two
tolerant-matching baselines (8-grams with one mismatch inside the decision list, and retrieval of reply windows by
sentence embeddings in the manner of Krishna et al., 2023).

The data readers, the student generator and the attribution are those of src/r11_benchmark.py: its definitions are
executed here, as in src/r12_benchmark.py, with new route branches inserted into the student generator. The routes of
R11 are unchanged, and the students of seeds 2100 and 2200 are rebuilt exactly.
Written after the analyses ran. No measure, class definition or parameter of the attribution changes.
Usage: python -I src/r15_benchmark.py DATA_DIR R8_TOKEN_CLASSES -> results/r15_benchmark.json"""
import sys, os, json, random, time
from collections import Counter
from difflib import SequenceMatcher
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'r11_benchmark.py')).read()
if len(sys.argv) > 2: src = src.replace("'results/flora_token_classes.csv.gz'", repr(sys.argv[2]))
cut = src.index("out = {'n_per_seed': 400}")
NEW = """        elif r == 'X-retype': type_in(passage(other_pool(XRUN, u), L), lab)
        elif r == 'X-chunks':
            blk = passage(other_pool(XRUN, u), L); i = 0
            while i < len(blk):
                k = R.randint(2, TAU); t += R.uniform(1, 20); paste(blk[i:i + k], lab); i += k
        elif r == 'X-mixed':
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri); ask(R.choice(PROM[u]), rep)
            a_, b_ = passage([rep], L), passage(other_pool(XRUN, u), L); blk, lb = [], []; i = j = 0; turn = 0
            while i < len(a_) or j < len(b_):
                k = R.randint(8, 30)
                if (turn % 2 == 0 and i < len(a_)) or j >= len(b_): blk += a_[i:i + k]; lb += ['C-mixed'] * len(a_[i:i + k]); i += k
                else: blk += b_[j:j + k]; lb += ['X-mixed'] * len(b_[j:j + k]); j += k
                turn += 1
            doc.extend(blk); truth.extend(lb); snap()
        elif r in ('A3-after', 'A3-regen'):
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri); ask(R.choice(PROM[u]), rep)
            blk = passage([rep], L); paste(blk if r == 'A3-after' else perturb(blk, .15), lab)
        elif r == 'A3-before':
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri)
            turns.append((t, R.choice(PROM[u]))); paste(passage([rep], L), lab)
            t += R.uniform(2, 60); replies.append((t, ' '.join(rep))); t += R.uniform(5, 120)
"""
anchor = "        elif r.startswith('E'):"
assert anchor in src
defs = src[:cut].replace(anchor, NEW + anchor)
defs = defs.replace("TRUTHMAP = {'M': 'T',", "TRUTHMAP = {'A3-after': 'X', 'A3-before': 'X', 'A3-regen': 'X', 'M': 'T',")
DATA_DIR = sys.argv[1]
sys.argv = [os.path.join(HERE, 'r11_benchmark.py'), DATA_DIR]
G_ = {'__name__': 'r11_defs', '__file__': os.path.join(HERE, 'r11_benchmark.py')}
exec(compile(defs, 'r11_benchmark.py', 'exec'), G_)
for k in ('make_student', 'BASE', 'EXT', 'attribute', 'scan', 'first_times', 'tokens', 'ngrams', 'N', 'TAU', 'ORDER',
          'counts', 'prf_from', 'bin_counts', 'bin_prf', 'student_err'):
    globals()[k] = G_[k]
Q = int(os.environ.get('QUICK', 0)); NS = 40 if Q else 400
t00 = time.perf_counter()
HARDX = BASE + [('X-retype', .05), ('X-chunks', .05), ('X-mixed', .05)]
A3 = BASE + [('A3-after', .05), ('A3-before', .05), ('A3-regen', .05)]
SETS = {}
R = random.Random(2100); SETS['base'] = [make_student(R, BASE) for _ in range(NS)]
R = random.Random(2200); SETS['ext'] = [make_student(R, EXT) for _ in range(NS)]
R = random.Random(2300); SETS['hard_x'] = [make_student(R, HARDX) for _ in range(NS)]
R = random.Random(2400); SETS['a3'] = [make_student(R, A3) for _ in range(NS)]
print('students built', time.perf_counter() - t00, flush=True)
out = {'n_per_set': NS, 'routes_added': {'hard_x': HARDX[-3:], 'a3': A3[-3:]}}


def big(x): return x is None or x > TAU


def token_iota(A, snaps):
    te, ins, last, tl, sid, seg = scan(snaps, N, [])
    io = [None] * len(A)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, A, last, autojunk=False).get_opcodes():
        if op == 'equal':
            for q in range(i2 - i1): io[i1 + q] = tl[j1 + q]
    return io


class Fuzzy(dict):
    """first time of each n-gram, matched exactly or with one token replaced (one wildcard position)"""
    def __init__(self, items, n):
        super().__init__()
        for t, text in sorted(items, key=lambda x: x[0]):
            for g in ngrams(tokens(text), n):
                for key in [g] + [g[:i] + ('\0',) + g[i + 1:] for i in range(n)]:
                    dict.setdefault(self, key, t)
    def _keys(self, g): return [g] + [g[:i] + ('\0',) + g[i + 1:] for i in range(len(g))]
    def __contains__(self, g): return any(dict.__contains__(self, k) for k in self._keys(g))
    def __getitem__(self, g): return min(dict.__getitem__(self, k) for k in self._keys(g) if dict.__contains__(self, k))


def attribute_fuzzy(st):
    orig = G_['first_times']
    G_['first_times'] = lambda items, n: Fuzzy(items, n)
    try: return attribute(st, st['snaps'], entry_rule='token')
    finally: G_['first_times'] = orig


# embedding retrieval over reply windows (Krishna et al., 2023, retrieval over the model's generations)
from sentence_transformers import SentenceTransformer
EMB = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
WIN, STRIDE = 20, 10
def windows(tk):
    if len(tk) <= WIN: return [(0, len(tk))] if tk else []
    st_ = list(range(0, len(tk) - WIN + 1, STRIDE))
    if st_[-1] != len(tk) - WIN: st_.append(len(tk) - WIN)
    return [(s, s + WIN) for s in st_]
CACHE = {}
def enc(txts):
    """embeddings of windows, cached by text (replies of a donor recur across simulated students)"""
    if not txts: return np.zeros((0, 384))
    new = [t for t in dict.fromkeys(txts) if t not in CACHE]
    if new:
        for t, v in zip(new, EMB.encode(new, batch_size=128, normalize_embeddings=True, show_progress_bar=False)): CACHE[t] = v
    return np.stack([CACHE[t] for t in txts])
def retrieval_scores(st):
    """per token: max cosine similarity of a covering window of A with any reply window (no direction), and the same
    with reply windows discarded when a prompt window sent earlier matches the A window at least as well"""
    A = tokens(st['final']); wa = windows(A)
    rw, rt = [], []
    for t, x in st['replies']:
        tk = tokens(x)
        for a, b in windows(tk): rw.append(' '.join(tk[a:b])); rt.append(t)
    pw, pt = [], []
    for t, x in st['turns']:
        tk = tokens(x)
        for a, b in windows(tk): pw.append(' '.join(tk[a:b])); pt.append(t)
    EA, ER, EP = enc([' '.join(A[a:b]) for a, b in wa]), enc(rw), enc(pw)
    s_any = np.zeros(len(A)); s_dir = np.zeros(len(A))
    if len(ER) and len(EA):
        SR = EA @ ER.T; SP = EA @ EP.T if len(EP) else np.zeros((len(EA), 0)); rt_, pt_ = np.array(rt), np.array(pt)
        for w_, (a, b) in enumerate(wa):
            best = SR[w_].max(); s_any[a:b] = np.maximum(s_any[a:b], best)
            j = int(SR[w_].argmax())
            earlier = SP[w_][pt_ < rt_[j]] if len(pt_) else np.zeros(0)
            d_ = 0.0 if (len(earlier) and earlier.max() >= best) else best
            s_dir[a:b] = np.maximum(s_dir[a:b], d_)
    return s_any, s_dir


def epa_rows(st, A, truthC, io, inrep):
    tb, tp = first_times(st['replies'], N), first_times(st['turns'], N); G = ngrams(A, N); te, *_ = scan(st['snaps'], N, G)
    auth = []
    for g in G:
        cand = [(tb[g], 0)] if g in tb else []
        if g in tp: cand.append((tp[g], 1))
        if g in te: cand.append((te[g], 1))
        auth.append(min(cand)[1] == 0 if cand else False)
    mk = [False] * len(A); nC = [0] * len(A); nW = [0] * len(A)
    for i, isC in enumerate(auth):
        for j in range(i, i + N):
            if isC: mk[j] = True; nC[j] += 1
            else: nW[j] += 1
    plain, iota = [], []
    for i, a in enumerate(truthC):
        conf = nC[i] if mk[i] else nW[i]
        plain.append((a, mk[i], conf)); iota.append((a, mk[i], -1 if (not mk[i] and big(io[i]) and inrep[i] == 0) else conf))
    return plain, iota


def rc_paper(rows):
    """convention of Section 4.3: abstentions decided last, scored as 'not chatbot text'; curve over coverage 0..1"""
    allt = sorted(rows, key=lambda x: -x[2]); n_ = len(allt)
    err = np.cumsum([a != b for a, b, _ in allt]) / np.arange(1, n_ + 1); cov = np.arange(1, n_ + 1) / n_
    return float(np.trapezoid(err, cov))


def rc_standard(rows, c_common=None):
    """El-Yaniv & Wiener (2010): risk over decided tokens only; the curve ends at the method's coverage.
    Returns coverage, selective risk at that coverage, and the area under the curve normalised by the coverage range,
    over the method's own range and over a common range [0, c_common]."""
    dec = sorted([(a, b, c) for a, b, c in rows if c != -1], key=lambda x: -x[2]); n_all = len(rows); n_ = len(dec)
    if not n_: return {'coverage': 0.0}
    err = np.cumsum([a != b for a, b, _ in dec]) / np.arange(1, n_ + 1); cov = np.arange(1, n_ + 1) / n_all
    o = {'coverage': float(cov[-1]), 'selective_risk': float(err[-1]), 'aurc_own_range': float(np.trapezoid(err, cov) / cov[-1])}
    if c_common:
        k = max(1, int(np.searchsorted(cov, c_common, side='right'))); o['aurc_common_range'] = float(np.trapezoid(err[:k], cov[:k]) / cov[k - 1])
        o['risk_at_common_coverage'] = float(err[k - 1])
    return o


def evaluate(name, STUD, with_retrieval=True):
    rows = {k: [] for k in ('ours', 'epa', 'epa_iota')}; per = []; byroute = {}; Ut, Ue = [], []
    bins = {k: [] for k in ('ours', 'fuzzy_attribution', 'retrieval_any_0.75', 'retrieval_any_0.85', 'retrieval_dir_0.75', 'retrieval_dir_0.85')}
    route_marks = {k: {} for k in bins}
    for st in STUD:
        A = tokens(st['final']); truthC = [x == 'C' for x in st['truth']]
        ot = attribute(st, st['snaps'], entry_rule='token'); cls = ot['cls']
        per.append(counts(st['truth'], cls)); Ut.append(st['truth'].count('C') / len(A)); Ue.append(cls.count('C') / len(A))
        for r, c in zip(st['route'], cls): byroute.setdefault(r, Counter())[c] += 1
        io = token_iota(A, st['snaps']); tbs = set(first_times(st['replies'], N)); inrep = [0] * len(A)
        for i, g in enumerate(ngrams(A, N)):
            if g in tbs:
                for j in range(i, i + N): inrep[j] += 1
        rows['ours'] += [(a, c == 'C', -1 if c == 'X' else ag) for a, c, ag in zip(truthC, cls, ot['agree'])]
        p_, i_ = epa_rows(st, A, truthC, io, inrep); rows['epa'] += p_; rows['epa_iota'] += i_
        marks = {'ours': [c == 'C' for c in cls], 'fuzzy_attribution': [c == 'C' for c in attribute_fuzzy(st)['cls']]}
        if with_retrieval:
            s_any, s_dir = retrieval_scores(st)
            for th in (.75, .85): marks[f'retrieval_any_{th}'] = list(s_any >= th); marks[f'retrieval_dir_{th}'] = list(s_dir >= th)
        for k, m in marks.items():
            bins[k].append(bin_counts(truthC, m))
            for r, x in zip(st['route'], m): route_marks[k].setdefault(r, Counter())[bool(x)] += 1
    o = {'per_class': {c: dict(zip(('precision', 'recall', 'f1'), prf_from(per, c))) for c in ORDER},
         'student_U': student_err(Ut, Ue),
         'route_to_class': {r: {k: v / sum(cc.values()) for k, v in cc.items()} for r, cc in byroute.items()}}
    o['chatbot_text_decision'] = {k: dict(zip(('precision', 'recall', 'f1'), bin_prf(v))) for k, v in bins.items() if v}
    o['marked_chatbot_by_route'] = {k: {r: c[True] / sum(c.values()) for r, c in v.items()} for k, v in route_marks.items() if v}
    o['aurc_paper_convention'] = {k: rc_paper(v) for k, v in rows.items()}
    c_common = min(rc_standard(v)['coverage'] for k, v in rows.items() if k != 'epa')
    o['selective_standard'] = {k: rc_standard(v, c_common) for k, v in rows.items()}
    o['selective_standard']['common_coverage'] = c_common
    print(name, json.dumps({k: o[k] for k in ('aurc_paper_convention', 'selective_standard')}, default=float), flush=True)
    print(name, {k: tuple(round(x, 3) for x in v.values()) for k, v in o['chatbot_text_decision'].items()}, flush=True)
    return o


for name, STUD in SETS.items():
    out[name] = evaluate(name, STUD)
    print(name, 'done', round(time.perf_counter() - t00), flush=True)
out['seconds'] = time.perf_counter() - t00
json.dump(out, open('results/r15_benchmark_quick.json' if Q else 'results/r15_benchmark.json', 'w'), indent=1, default=float)
print('done', out['seconds'])
