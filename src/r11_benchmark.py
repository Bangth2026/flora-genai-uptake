"""Analysis log R11, P-1 to P-3 (benchmark part) and B-4 to B-9: extended benchmark with planted provenance.

Routes of R10 plus six routes outside the specification, random route mixes, repeated seeds, student bootstrap
intervals, earliest-plausible and paste-log baselines, n-gram and token entry rules, and risk-coverage curves.
Text is real FLoRA text; edit sizes and intervals are fitted to the FLoRA writing log exactly as in R10.
Usage: python -I src/r11_benchmark.py DATA_DIR [WORK_DIR] -> results/r11_benchmark.json
WORK_DIR receives the simulated students of seed 2100 (pickle) for the LLM baseline (src/r11_llm_baseline.py)."""
import sys, os, json, random, time, pickle, hashlib
from collections import Counter
from difflib import SequenceMatcher
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demo'))
from provenance import tokens, ngrams, ORDER
from baselines import winnow_mark, tile_mark, overlap_mark
from provenance_online import OnlineAttributor

D = sys.argv[1]; WORK = sys.argv[2] if len(sys.argv) > 2 else None
N = 8; TAU = 5; NB = 2000
Q = int(os.environ.get('QUICK', 0))  # QUICK=1: small sizes for testing only
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
an = pd.read_pickle(f'{D}/annot.pkl')
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])

# ---------- empirical distributions from the FLoRA writing log (60 random students), identical to R10
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms')
us = sorted(w.user_id.unique()); rs = np.random.default_rng(2100); pick = set(rs.choice(us, 60, replace=False))
small_sz, small_dt, large_sz, large_gap = [], [], [], []
for u, g in w[w.user_id.isin(pick)].sort_values(['user_id', 't']).groupby('user_id'):
    prev = []; tp = None; tl = None
    for t, c in zip(g.t, g.writing_content):
        cur = tokens(c); m = min(len(prev), len(cur)); pre = 0
        while pre < m and prev[pre] == cur[pre]: pre += 1
        suf = 0
        while suf < m - pre and prev[-1 - suf] == cur[-1 - suf]: suf += 1
        ins = len(cur) - pre - suf
        if tp is not None and ins > 0:
            dt = (t - tp).total_seconds()
            if ins <= TAU and dt < 60: small_sz.append(ins); small_dt.append(dt)
            elif ins > TAU:
                large_sz.append(ins)
                if tl is not None: large_gap.append((t - tl).total_seconds())
                tl = t
        prev, tp = cur, t
del w
large_sz = [x for x in large_sz if 12 <= x <= 300]; large_gap = [x for x in large_gap if 5 <= x <= 3600]

REPL = {u: [tokens(s) for s in g.chatbot_utterance_text.astype(str)] for u, g in d.groupby('user_id')}
PROM = {u: [str(s) for s in g.user_utterance_text.astype(str)] for u, g in d.groupby('user_id')}
DON = [u for u in REPL if sum(len(r) for r in REPL[u]) >= 400]
def runs(cls, minlen=15):
    out = {}
    for u, g in tc.groupby('user_id'):
        tk, cc = g.token.astype(str).tolist(), g['class'].tolist(); i = 0
        while i < len(cc):
            j = i
            while j < len(cc) and cc[j] == cc[i]: j += 1
            if cc[i] == cls and j - i >= minlen: out.setdefault(u, []).append(tk[i:j])
            i = j
    return out
TRUN, XRUN = runs('T'), runs('X')
HL = [tokens(s) for s in an.highlight_text.dropna().astype(str)]; HL = [h for h in HL if len(h) >= 10]
SHARED_TXT = [' '.join(h) for h in HL]
SHG = {g for s in SHARED_TXT for g in ngrams(tokens(s), N)}
VOC = sorted({x for v in TRUN.values() for s in v for x in s})

BASE = [('T', .14), ('M', .04), ('C-paste', .14), ('C-retype', .04), ('C-para10', .04), ('C-para25', .03),
        ('E-prompt', .07), ('E-editor', .03), ('E-editor-auto', .03), ('E-deleted', .03), ('S', .03), ('X-ai', .14), ('X-other', .25)]
OFFSPEC = [('C-edit20', .03), ('C-edit50', .03), ('C-interleave', .03), ('C-regen', .03), ('C-regen-unlogged', .03), ('C-stream', .03), ('U-undo', .03)]
EXT = BASE + OFFSPEC
TRUTHMAP = {'M': 'T', 'Tedit': 'T', 'Tint': 'T', 'U-undo-T': 'T', 'U-undo-C': 'C'}
def TRUTH(r): return TRUTHMAP.get(r, r[0])

def make_student(R, routes):
    def passage(pool, L):
        seq = R.choice(pool)
        if len(seq) <= L: return list(seq)
        s = R.randrange(0, len(seq) - L + 1); return list(seq[s:s + L])
    def other_pool(dic, u):
        keys = [k for k in dic if k != u]; return dic[R.choice(keys)]
    u = R.choice(DON); replies, turns, snaps, doc, truth = [], [], [], [], []
    t = 0.0
    def snap(): snaps.append((t, ' '.join(doc)))
    def type_in(blk, lab, at=None):
        nonlocal t
        pos = len(doc) if at is None else at; i = 0
        while i < len(blk):
            k = R.choice(small_sz); t += R.choice(small_dt)
            doc[pos:pos] = blk[i:i + k]; truth[pos:pos] = [lab] * len(blk[i:i + k]); pos += len(blk[i:i + k]); i += k; snap()
    def paste(blk, lab):
        doc.extend(blk); truth.extend([lab] * len(blk)); snap()
    def ask(prompt_text, reply_toks, log_reply=True):
        nonlocal t
        turns.append((t, prompt_text)); t += R.uniform(5, 30)
        if log_reply: replies.append((t, ' '.join(reply_toks)))
        t += R.uniform(5, 300)
    def perturb(blk, p): return [R.choice(VOC) if R.random() < p else x for x in blk]
    nblk = R.randint(6, 14); used = set()
    names, wts = [a for a, _ in routes], [b for _, b in routes]
    for _ in range(nblk):
        r = R.choices(names, wts)[0]; L = R.choice(large_sz); t += R.choice(large_gap); lab = r
        if r == 'T': type_in(passage(other_pool(TRUN, u), L), lab)
        elif r == 'M':
            blk = passage(other_pool(TRUN, u), L); start = len(doc); type_in(blk, lab)
            if start > 0:
                t += R.choice(large_gap); seg, lb = doc[start:], truth[start:]; del doc[start:]; del truth[start:]; snap()
                at = R.randrange(0, start + 1); t += R.uniform(1, 10); doc[at:at] = seg; truth[at:at] = lb; snap()
        elif r in ('C-paste', 'C-retype', 'C-para10', 'C-para25'):
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri)
            blk = passage([rep], L); ask(R.choice(PROM[u]), rep)
            if r == 'C-paste': paste(blk, lab)
            elif r == 'C-retype': type_in(blk, lab)
            else: paste(perturb(blk, .10 if r == 'C-para10' else .25), lab)
        elif r in ('C-edit20', 'C-edit50'):
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri)
            blk = passage([rep], L); ask(R.choice(PROM[u]), rep); start = len(doc); paste(blk, lab)
            p = .20 if r == 'C-edit20' else .50; pos = R.sample(range(start, start + len(blk)), int(round(p * len(blk))))
            t += R.uniform(10, 300)
            for q in sorted(pos):
                t += R.choice(small_dt); doc[q] = R.choice(VOC); truth[q] = 'Tedit'; snap()
        elif r == 'C-interleave':
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri)
            blk = passage([rep], L); own = passage(other_pool(TRUN, u), L); ask(R.choice(PROM[u]), rep)
            i = j = 0; turn = 0
            while i < len(blk):
                k = R.randint(8, 30)
                if turn % 2 == 0: paste(blk[i:i + k], lab); i += k
                else:
                    if j >= len(own): j = 0
                    type_in(own[j:j + k], 'Tint'); j += k
                turn += 1
        elif r in ('C-regen', 'C-regen-unlogged'):
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri)
            ask(R.choice(PROM[u]), rep)
            rep2 = perturb(rep, .15); t += R.uniform(5, 30)
            if r == 'C-regen': replies.append((t, ' '.join(rep2)))
            t += R.uniform(5, 120); paste(passage([rep2], L), lab)
        elif r == 'C-stream':
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri)
            turns.append((t, R.choice(PROM[u]))); t += R.uniform(3, 10)
            paste(passage([rep], L), lab); lag = R.uniform(2, 20)
            replies.append((t + lag, ' '.join(rep))); t += lag + R.uniform(5, 120)
        elif r == 'U-undo':
            start = len(doc)
            if R.random() < .5:
                blk = passage(other_pool(TRUN, u), L); type_in(blk, 'U-undo-T')
            else:
                ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri)
                blk = passage([rep], L); ask(R.choice(PROM[u]), rep); paste(blk, 'U-undo-C')
            seg, lb = doc[start:], truth[start:]; t += R.uniform(2, 30); del doc[start:]; del truth[start:]; snap()
            t += R.uniform(5, 60); doc.extend(seg); truth.extend(lb); snap()
        elif r.startswith('E'):
            blk = passage(other_pool(TRUN, u), L); pre = passage([R.choice(REPL[u])], 20); post = passage([R.choice(REPL[u])], 20)
            if r == 'E-prompt':
                ask('improve this ' + ' '.join(blk), pre + blk + post); paste(blk, lab)
            elif r == 'E-editor':
                type_in(blk, lab); t += R.choice(large_gap); ask('check this ' + ' '.join(blk), pre + blk + post)
            elif r == 'E-editor-auto':
                type_in(blk, lab); t += R.choice(large_gap); ask(R.choice(PROM[u]), pre + blk + post)
            else:
                start = len(doc); type_in(blk, 'tmp'); del doc[start:]; del truth[start:]; snap(); t += R.uniform(10, 120)
                ask('fix this ' + ' '.join(blk), pre + blk + post); paste(blk, lab)
        elif r == 'S': paste(list(R.choice(HL)), lab)
        elif r == 'X-ai': paste(passage(other_pool(REPL, u), L), lab)
        else: paste(passage(other_pool(XRUN, u), L), lab)
    T_end = t
    for ri, rep in enumerate(REPL[u]):
        if ri in used: continue
        tt = R.uniform(0, T_end); turns.append((tt, R.choice(PROM[u]))); replies.append((tt + 10, ' '.join(rep)))
    return dict(final=' '.join(doc), truth=[TRUTH(x) for x in truth], route=list(truth), replies=sorted(replies), turns=sorted(turns), snaps=snaps, donor=int(u))

# ---------- attribution
def first_times(items, n):
    f = {}
    for t, text in sorted(items, key=lambda x: x[0]):
        for g in ngrams(tokens(text), n): f.setdefault(g, t)
    return f

def scan(snaps, n, need):
    """n-gram entry (R10): first snapshot containing each n-gram and the tokens inserted in that transition.
    Token entry (R9 F-8): size and snapshot index of the insertion that brought in each token of the last snapshot."""
    need = set(need); te, ins = {}, {}; prev = []; lab = []; sid = []; seg = []
    for k, (t, text) in enumerate(snaps):
        cur = tokens(text); m = min(len(prev), len(cur)); pre = 0
        while pre < m and prev[pre] == cur[pre]: pre += 1
        suf = 0
        while suf < m - pre and prev[-1 - suf] == cur[-1 - suf]: suf += 1
        a, b = prev[pre:len(prev) - suf], cur[pre:len(cur) - suf]
        if not a and not b: prev = cur; continue
        ops = [('insert', 0, 0, 0, len(b))] if (b and not a) else (SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if b else [])
        tot = len(b) if (b and (not a or len(b) == 1)) else sum(j2 - j1 for op, i1, i2, j1, j2 in ops if op in ('insert', 'replace'))
        if b and need:
            lo, hi = max(0, pre - n + 1), min(len(cur) - n + 1, len(cur) - suf)
            for s in range(lo, max(lo, hi)):
                g = tuple(cur[s:s + n])
                if g in need: te[g] = t; ins[g] = tot; need.discard(g)
        nl = lab[:pre]; ns = sid[:pre]; nseg = seg[:pre]; mid_l, mid_s, mid_g = [None] * len(b), [None] * len(b), [None] * len(b)
        for op, i1, i2, j1, j2 in ops:
            for q in range(j2 - j1):
                if op == 'equal': mid_l[j1 + q] = lab[pre + i1 + q]; mid_s[j1 + q] = sid[pre + i1 + q]; mid_g[j1 + q] = seg[pre + i1 + q]
                else: mid_l[j1 + q] = j2 - j1; mid_s[j1 + q] = k; mid_g[j1 + q] = (k, j1)
        lab = nl + mid_l + (lab[len(prev) - suf:] if suf else []); sid = ns + mid_s + (sid[len(prev) - suf:] if suf else [])
        seg = nseg + mid_g + (seg[len(prev) - suf:] if suf else [])
        prev = cur
    return te, ins, prev, lab, sid, seg

def attribute(st, snaps, n=N, tau=TAU, delta=0.0, entry_rule='ngram', keep=False):
    A = tokens(st['final']); G = ngrams(A, n); tb, tp = first_times(st['replies'], n), first_times(st['turns'], n)
    te, ins, last, tl, sid, seg = scan(snaps, n, G); lab = {}
    for g in set(G):
        bot = g in tb; pf = bot and g in tp and tp[g] < tb[g]; sh = g in SHG
        if bot and not pf and not sh and g in te and tb[g] <= te[g] + delta: lab[g] = 'C'
        elif bot and (pf or (g in te and te[g] < tb[g])): lab[g] = 'E'
        elif sh: lab[g] = 'S'
        elif g in te: lab[g] = 'T' if ins[g] <= tau else 'X'
        else: lab[g] = 'X'
    rank = {c: i for i, c in enumerate(ORDER)}; best = [9] * len(A); agree = [0] * len(A)
    for i, g in enumerate(G):
        for j in range(i, i + n): best[j] = min(best[j], rank[lab[g]])
    cls = [ORDER[b] if b < 9 else 'X' for b in best]
    for i, g in enumerate(G):
        for j in range(i, i + n):
            if lab[g] == cls[j]: agree[j] += 1
    if entry_rule == 'token':
        tok_ins = [None] * len(A)
        for op, i1, i2, j1, j2 in SequenceMatcher(None, A, last, autojunk=False).get_opcodes():
            if op == 'equal':
                for q in range(i2 - i1): tok_ins[i1 + q] = tl[j1 + q]
        cls = [c if c in 'CES' else ('T' if (tok_ins[i] is not None and tok_ins[i] <= tau) else 'X') for i, c in enumerate(cls)]
    out = dict(A=A, cls=cls, agree=agree)
    if keep: out.update(lab=lab, tb=tb, tp=tp, te=te, ins=ins, last=last, tl=tl, sid=sid, seg=seg)
    return out

def thin(snaps, s_):
    if not s_: return snaps
    kept, last = [], -1e18
    for x in snaps:
        if x[0] - last >= s_: kept.append(x); last = x[0]
    if kept[-1] is not snaps[-1]: kept.append(snaps[-1])
    return kept

# ---------- scores
def counts(truth, pred):
    """per-class tp, predicted, actual for one student"""
    return {c: (sum(a == c and b == c for a, b in zip(truth, pred)), sum(b == c for b in pred), sum(a == c for a in truth)) for c in ORDER}
def prf_from(cnts, c):
    tp_ = sum(x[c][0] for x in cnts); pp = sum(x[c][1] for x in cnts); ap = sum(x[c][2] for x in cnts)
    p = tp_ / pp if pp else np.nan; r = tp_ / ap if ap else np.nan
    return p, r, (2 * p * r / (p + r)) if (p + r) > 0 else np.nan
def bin_counts(truthC, mark): return (sum(a and b for a, b in zip(truthC, mark)), sum(mark), sum(truthC))
def bin_prf(bc):
    tp_ = sum(x[0] for x in bc); pp = sum(x[1] for x in bc); ap = sum(x[2] for x in bc)
    p = tp_ / pp if pp else np.nan; r = tp_ / ap if ap else np.nan
    return p, r, (2 * p * r / (p + r)) if (p + r) > 0 else np.nan
def student_err(true_sh, est_sh):
    a, b = np.array(true_sh), np.array(est_sh)
    return {'mae': float(np.mean(np.abs(b - a))), 'bias': float(np.mean(b - a)), 'spearman': float(stats.spearmanr(a, b)[0]),
            'agree_ge50': float(np.mean((a >= .5) == (b >= .5)))}
def boot(per_student, fn, rng, nb=None):
    nb = nb or (50 if Q else NB)
    k = len(per_student); vals = []
    for _ in range(nb):
        idx = rng.integers(0, k, k); vals.append(fn([per_student[i] for i in idx]))
    vals = np.array(vals, dtype=float); return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

def run_conditions(STUD, conds):
    res = {}
    for name, s_, tau, rule, delta in conds:
        cnts, acc, Ut, Ue, byroute = [], [], [], [], {}
        for st in STUD:
            o = attribute(st, thin(st['snaps'], s_), tau=tau, entry_rule=rule, delta=delta); cls = o['cls']
            cnts.append(counts(st['truth'], cls)); acc.append(sum(a == b for a, b in zip(st['truth'], cls)) / len(cls))
            Ut.append(st['truth'].count('C') / len(cls)); Ue.append(cls.count('C') / len(cls))
            for r, c in zip(st['route'], cls): byroute.setdefault(r, Counter())[c] += 1
        tot = sum(sum(x[c][2] for c in ORDER) for x in cnts); corr = sum(sum(x[c][0] for c in ORDER) for x in cnts)
        res[name] = {'accuracy': corr / tot, 'per_class': {c: dict(zip(('precision', 'recall', 'f1'), prf_from(cnts, c))) for c in ORDER},
                     'student_Ustar': student_err(Ut, Ue), 'route_to_class': {r: {k: v / sum(cc.values()) for k, v in cc.items()} for r, cc in byroute.items()}}
        print(name, round(res[name]['accuracy'], 3), {c: tuple(round(x, 3) for x in prf_from(cnts, c)) for c in ORDER}, flush=True)
    return res

out = {'n_per_seed': 400}
t00 = time.perf_counter()
# ---------- B-6 seeds (R10 routes, base mix) with the primary token entry rule and the R10 n-gram entry rule
SEEDS = list(range(2100, 2102 if Q else 2120))
seed_res = {'ngram': [], 'token': []}
for sd in SEEDS:
    R = random.Random(sd); STUD = [make_student(R, BASE) for _ in range(40 if Q else 400)]
    for rule in ('ngram', 'token'):
        cnts, acc = [], []
        for st in STUD:
            cls = attribute(st, st['snaps'], entry_rule=rule)['cls']; cnts.append(counts(st['truth'], cls))
        tot = sum(sum(x[c][2] for c in ORDER) for x in cnts); corr = sum(sum(x[c][0] for c in ORDER) for x in cnts)
        seed_res[rule].append({'accuracy': corr / tot, **{f'{c}_{m}': v for c in ORDER for m, v in zip(('p', 'r', 'f1'), prf_from(cnts, c))}})
    print('seed', sd, round(seed_res['token'][-1]['accuracy'], 3), round(seed_res['token'][-1]['C_f1'], 3), flush=True)
out['B6_seeds'] = {rule: {k: {'mean': float(np.mean([s[k] for s in v])), 'lo': float(np.percentile([s[k] for s in v], 2.5)), 'hi': float(np.percentile([s[k] for s in v], 97.5))}
                          for k in v[0]} for rule, v in seed_res.items()}
print('seeds done', time.perf_counter() - t00, flush=True)

# ---------- seed 2100 (identical students to R10): conditions, bootstrap, baselines, risk-coverage, properties
R = random.Random(2100); STUD = [make_student(R, BASE) for _ in range(40 if Q else 400)]
R = random.Random(2200); STUDX = [make_student(R, EXT) for _ in range(40 if Q else 400)]
if WORK:
    os.makedirs(WORK, exist_ok=True); pickle.dump({'base': STUD, 'ext': STUDX, 'SHARED_TXT': SHARED_TXT}, open(f'{WORK}/r11_students.pkl', 'wb'))
CONDS = [('full_ngram', 0, 5, 'ngram', 0), ('full_token', 0, 5, 'token', 0)]
for s_ in (5, 15, 60):
    for tau in (5, 20):
        for rule in ('ngram', 'token'): CONDS.append((f'thin{s_}_tau{tau}_{rule}', s_, tau, rule, 0))
out['B8_entry_rule_base'] = run_conditions(STUD, CONDS)
out['B4_offspec'] = run_conditions(STUDX, [('full_ngram', 0, 5, 'ngram', 0), ('full_token', 0, 5, 'token', 0), ('full_token_delta30', 0, 5, 'token', 30.0)])
print('conditions done', time.perf_counter() - t00, flush=True)

# baselines on seed 2100 (base routes) and on the extended routes
def epa_mark(st, A):
    """earliest plausible attribution over revisions (turns: student, replies: chatbot, snapshots: student)."""
    tb, tp = first_times(st['replies'], N), first_times(st['turns'], N); G = ngrams(A, N); te, *_ = scan(st['snaps'], N, G)
    m = [False] * len(A)
    for i, g in enumerate(G):
        cand = [(tb[g], 0)] if g in tb else []
        if g in tp: cand.append((tp[g], 1))
        if g in te: cand.append((te[g], 1))
        if cand and min(cand)[1] == 0:
            for j in range(i, i + N): m[j] = True
    return m
def paste_mark(st, A):
    """paste-log rule: tokens entering in an insertion > tau are chatbot text if >= half of that insertion's 8-grams
    occur in an earlier reply; returns (chatbot mark, typed mark)."""
    te, ins, last, tl, sid, seg = scan(st['snaps'], N, [])
    segtoks = {}
    for i, sg in enumerate(seg):
        if sg is not None: segtoks.setdefault(sg, []).append(last[i])
    stime = {k: t for k, (t, _) in enumerate(st['snaps'])}
    rep = [(t, set(ngrams(tokens(x), N))) for t, x in st['replies']]; segC = {}
    for sg, tk in segtoks.items():
        gs = ngrams(tk, N)
        if len(tk) <= TAU or not gs: segC[sg] = False; continue
        prior = set().union(*[s for t, s in rep if t <= stime[sg[0]]]) if rep else set()
        segC[sg] = np.mean([g in prior for g in gs]) >= .5
    mapC = [False] * len(A); mapT = [False] * len(A)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, A, last, autojunk=False).get_opcodes():
        if op == 'equal':
            for q in range(i2 - i1):
                sg = seg[j1 + q]; mapC[i1 + q] = bool(segC.get(sg, False)); mapT[i1 + q] = tl[j1 + q] is not None and tl[j1 + q] <= TAU
    return mapC, mapT

def baselines(STUD_, rng, tag):
    keys = ('ours_token', 'ours_ngram', 'online', 'earliest_plausible', 'paste_log', 'dialogue_only', 'direction_blind', 'naive_overlap', 'winnowing', 'string_tiling')
    per = {k: [] for k in keys}; sh = {k: ([], []) for k in keys}; rc = {k: [] for k in ('ours_token', 'dialogue_only', 'naive_overlap', 'earliest_plausible')}
    trans = Counter(); p1_viol = 0; p1_n = 0; pl_route = {}
    for st in STUD_:
        A = tokens(st['final']); truthC = [x == 'C' for x in st['truth']]; rep = [tokens(x) for _, x in st['replies']]; pro = [tokens(x) for _, x in st['turns']]
        tb, tp = first_times(st['replies'], N), first_times(st['turns'], N); prg = {g for p_ in pro for g in ngrams(p_, N)}
        ot = attribute(st, st['snaps'], entry_rule='token', keep=True); on = attribute(st, st['snaps'], entry_rule='ngram')
        # P-1 soundness of C
        for g, l in ot['lab'].items():
            if g in ot['te'] and g in tb and ot['te'][g] < tb[g]:
                p1_n += 1; p1_viol += l == 'C'
        att = OnlineAttributor(shared=SHARED_TXT, n=N, tau=TAU); first_lab = {}
        ev = [(t, 0, x) for t, x in st['turns']] + [(t, 1, x) for t, x in st['replies']] + [(t, 2, x) for t, x in st['snaps']]
        for t, k, x in sorted(ev, key=lambda e: (e[0], e[1])):
            if k == 0: att.prompt(t, x)
            elif k == 1: att.reply(t, x)
            else:
                for g, l in att.snapshot(t, x).items(): first_lab.setdefault(g, l)
        for g, l in first_lab.items(): trans[(l, att.label[g])] += 1
        oc = att.classify(st['final'])[1]
        def mark_from(gs):
            S_ = set(gs); m = [False] * len(A)
            for i, g in enumerate(ngrams(A, N)):
                if g in S_:
                    for j in range(i, i + N): m[j] = True
            return m
        pm = paste_mark(st, A)
        marks = {'ours_token': [c == 'C' for c in ot['cls']], 'ours_ngram': [c == 'C' for c in on['cls']], 'online': [c == 'C' for c in oc],
                 'earliest_plausible': epa_mark(st, A), 'paste_log': pm[0],
                 'dialogue_only': mark_from(g for g in tb if not (g in tp and tp[g] < tb[g])),
                 'direction_blind': mark_from(g for r_ in rep for g in ngrams(r_, N) if g not in prg), 'naive_overlap': overlap_mark(A, rep, N),
                 'winnowing': winnow_mark(A, rep, 8, 4), 'string_tiling': tile_mark(A, rep, 8)}
        for r_, m_ in zip(st['route'], pm[0]): pl_route.setdefault(r_, Counter())[m_] += 1
        for k, m in marks.items():
            per[k].append(bin_counts(truthC, m)); sh[k][0].append(np.mean(truthC)); sh[k][1].append(np.mean(m))
        # risk-coverage material: (truthC, predC, confidence)
        rc['ours_token'].append([(a, c == 'C', -1 if c == 'X' else ag) for a, c, ag in zip(truthC, ot['cls'], ot['agree'])])
        for k, gs in (('dialogue_only', {g for g in tb if not (g in tp and tp[g] < tb[g])}), ('naive_overlap', set(tb))):
            cov = [0] * len(A); hit = [0] * len(A)
            for i, g in enumerate(ngrams(A, N)):
                for j in range(i, i + N): cov[j] += 1; hit[j] += g in gs
            s = [h / c if c else 0 for h, c in zip(hit, cov)]
            rc[k].append([(a, x >= .5, abs(x - .5)) for a, x in zip(truthC, s)])
        em = marks['earliest_plausible']; rc['earliest_plausible'].append([(a, b, 1) for a, b in zip(truthC, em)])
    res = {}
    for k in keys:
        p, r, f = bin_prf(per[k])
        res[k] = {'precision': p, 'recall': r, 'f1': f,
                  'ci_precision': boot(per[k], lambda x: bin_prf(x)[0], rng), 'ci_recall': boot(per[k], lambda x: bin_prf(x)[1], rng), 'ci_f1': boot(per[k], lambda x: bin_prf(x)[2], rng),
                  'student_share': student_err(*sh[k])}
        if k != 'ours_token':
            pair = list(zip(per['ours_token'], per[k]))
            res[k]['f1_diff_ours_minus_this'] = bin_prf(per['ours_token'])[2] - f
            res[k]['ci_f1_diff'] = boot(pair, lambda x: bin_prf([a for a, _ in x])[2] - bin_prf([b for _, b in x])[2], rng)
        print(tag, k, round(p, 3), round(r, 3), round(f, 3), res[k]['ci_f1'], res[k].get('ci_f1_diff'), flush=True)
    # risk-coverage
    RC = {}
    for k, v in rc.items():
        allt = [x for s_ in v for x in s_]; allt.sort(key=lambda x: -x[2]); n_ = len(allt)
        err = np.cumsum([a != b for a, b, _ in allt]) / np.arange(1, n_ + 1); cov = np.arange(1, n_ + 1) / n_
        RC[k] = {'aurc': float(np.trapezoid(err, cov)), 'risk_at': {str(c): float(err[int(c * n_) - 1]) for c in (.25, .5, .6, .7, .8, .9, 1.0)},
                 'curve': [[float(cov[i]), float(err[i])] for i in np.linspace(0, n_ - 1, 50).astype(int)]}
    ourscov = np.mean([c != -1 for s_ in rc['ours_token'] for _, _, c in s_])
    RC['ours_token']['coverage_without_X'] = float(ourscov)
    RC['ours_token']['risk_without_X'] = float(np.mean([a != b for s_ in rc['ours_token'] for a, b, c in s_ if c != -1]))
    res['paste_log']['marked_share_by_route'] = {r_: c[True] / sum(c.values()) for r_, c in pl_route.items()}
    return res, RC, {f'{a}->{b}': v for (a, b), v in trans.items()}, {'n_editor_first_ngrams': p1_n, 'labelled_C': p1_viol}

rng = np.random.default_rng(2110)
out['B7_baselines_base'], out['B9_risk_coverage_base'], out['P2_online_transitions_base'], out['P1_soundness_base'] = baselines(STUD, rng, 'base')
out['B7_baselines_ext'], out['B9_risk_coverage_ext'], out['P2_online_transitions_ext'], out['P1_soundness_ext'] = baselines(STUDX, rng, 'ext')
# per-class bootstrap intervals, seed 2100, token rule
per = [counts(st['truth'], attribute(st, st['snaps'], entry_rule='token')['cls']) for st in STUD]
out['B6_bootstrap_per_class_seed2100'] = {c: {m: boot(per, (lambda x, c=c, i=i: prf_from(x, c)[i]), rng, 1000) for i, m in enumerate(('precision', 'recall', 'f1'))} for c in ORDER}
# P-3 determinism
h = lambda: hashlib.sha256(json.dumps([attribute(st, st['snaps'], entry_rule='token')['cls'] for st in STUD[:100]]).encode()).hexdigest()
out['P3_determinism'] = {'run1': h(), 'run2': h()}; out['P3_determinism']['identical'] = out['P3_determinism']['run1'] == out['P3_determinism']['run2']
print('baselines done', time.perf_counter() - t00, flush=True)

# ---------- B-5 route mix
MIX = []; rmix = np.random.default_rng(2120)
for k in range(2 if Q else 50):
    for routes, tag in ((BASE, 'base'), (EXT, 'ext')):
        wv = np.array([b for _, b in routes]); wv = wv / wv.sum(); dr = rmix.dirichlet(13 * wv)
        R = random.Random(3000 + k); S_ = [make_student(R, list(zip([a for a, _ in routes], dr))) for _ in range(20 if Q else 200)]
        cnts, Ut, Ue, acc = [], [], [], []
        for st in S_:
            cls = attribute(st, st['snaps'], entry_rule='token')['cls']; cnts.append(counts(st['truth'], cls))
            Ut.append(st['truth'].count('C') / len(cls)); Ue.append(cls.count('C') / len(cls))
        tot = sum(sum(x[c][2] for c in ORDER) for x in cnts); corr = sum(sum(x[c][0] for c in ORDER) for x in cnts)
        p, r, f = prf_from(cnts, 'C'); se = student_err(Ut, Ue)
        MIX.append({'mix': k, 'routes': tag, 'accuracy': corr / tot, 'C_p': p, 'C_r': r, 'C_f1': f, 'U_mae': se['mae'], 'U_spearman': se['spearman'], 'agree_ge50': se['agree_ge50']})
    print('mix', k, MIX[-2]['C_f1'], MIX[-1]['C_f1'], flush=True)
mx = pd.DataFrame(MIX)
out['B5_route_mix'] = {tag: {c: {'median': float(g[c].median()), 'p5': float(g[c].quantile(.05)), 'p95': float(g[c].quantile(.95)), 'min': float(g[c].min())}
                             for c in ('accuracy', 'C_p', 'C_r', 'C_f1', 'U_mae', 'U_spearman', 'agree_ge50')} for tag, g in mx.groupby('routes')}
out['B5_dirichlet_concentration'] = 13
out['seconds'] = time.perf_counter() - t00
json.dump(out, open('results/r11_benchmark_quick.json' if Q else 'results/r11_benchmark.json', 'w'), indent=1, default=float)
print('done', out['seconds'])
