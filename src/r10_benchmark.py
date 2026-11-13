"""Analysis log R10, B-1 to B-3: benchmark with planted provenance, fitted to FLoRA, and baselines on the same logs.

Simulated students borrow the chatbot replies and prompts of one real FLoRA student and write documents from blocks
with known routes (see ANALYSIS_LOG_R10.md). Text is real FLoRA text; edit sizes and intervals are drawn from the
FLoRA writing log. The rules are those of demo/provenance.py; editor entry uses the prefix/suffix trimming of the
study code (src/r9_flora.py) before aligning the changed region.
Usage: python -I src/r10_benchmark.py DATA_DIR -> results/r10_benchmark.json"""
import sys, os, json, random, time
from collections import Counter
from difflib import SequenceMatcher
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demo'))
from provenance import tokens, ngrams, ORDER
from baselines import winnow_mark, tile_mark, overlap_mark
from provenance_online import OnlineAttributor

D = sys.argv[1]; R = random.Random(2100); NSTUD = int(os.environ.get('NSTUD', 400)); N = 8; TAU = 5
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
an = pd.read_pickle(f'{D}/annot.pkl')
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])

# ---------- empirical distributions from the FLoRA writing log (60 random students)
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
FIT = {'small_insert_sizes_quartiles': np.percentile(small_sz, [25, 50, 75]).tolist(), 'small_insert_interval_s_quartiles': np.percentile(small_dt, [25, 50, 75]).tolist(),
       'large_insert_sizes_quartiles': np.percentile(large_sz, [25, 50, 75]).tolist(), 'gap_between_large_inserts_s_quartiles': np.percentile(large_gap, [25, 50, 75]).tolist(),
       'n_small': len(small_sz), 'n_large': len(large_sz)}
print('fit', FIT, flush=True)
PL = [len(tokens(s)) for s in d.user_utterance_text.astype(str)]
FIT['prompt_tokens_quartiles'] = np.percentile(PL, [25, 50, 75]).tolist()

# ---------- text pools
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

def passage(pool, L):
    seq = R.choice(pool)
    if len(seq) <= L: return list(seq)
    s = R.randrange(0, len(seq) - L + 1); return list(seq[s:s + L])
def other_pool(dic, u):
    keys = [k for k in dic if k != u]; return dic[R.choice(keys)]

ROUTES = [('T', .14), ('M', .04), ('C-paste', .14), ('C-retype', .04), ('C-para10', .04), ('C-para25', .03),
          ('E-prompt', .07), ('E-editor', .03), ('E-editor-auto', .03), ('E-deleted', .03), ('S', .03), ('X-ai', .14), ('X-other', .25)]
TRUTH = lambda r: 'T' if r in ('T', 'M') else r[0]
VOC = sorted({x for v in TRUN.values() for s in v for x in s})

def student():
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
        nonlocal t
        doc.extend(blk); truth.extend([lab] * len(blk)); snap()
    def ask(prompt_text, reply_toks):
        nonlocal t
        turns.append((t, prompt_text)); t += R.uniform(5, 30); replies.append((t, ' '.join(reply_toks))); t += R.uniform(5, 300)
    nblk = R.randint(6, 14); used = set()
    for _ in range(nblk):
        r = R.choices([a for a, _ in ROUTES], [b for _, b in ROUTES])[0]; L = R.choice(large_sz); t += R.choice(large_gap)
        lab = r
        if r == 'T': type_in(passage(other_pool(TRUN, u), L), lab)
        elif r == 'M':
            blk = passage(other_pool(TRUN, u), L); start = len(doc); type_in(blk, lab); n_ = len(doc) - start
            if start > 0:
                t += R.choice(large_gap); seg, lb = doc[start:], truth[start:]; del doc[start:]; del truth[start:]; snap()
                at = R.randrange(0, start + 1); t += R.uniform(1, 10); doc[at:at] = seg; truth[at:at] = lb; snap()
        elif r.startswith('C'):
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri)
            blk = passage([rep], L); ask(R.choice(PROM[u]), rep)
            if r == 'C-paste': paste(blk, lab)
            elif r == 'C-retype': type_in(blk, lab)
            else:
                p = .10 if r == 'C-para10' else .25
                paste([R.choice(VOC) if R.random() < p else x for x in blk], lab)
        elif r.startswith('E'):
            blk = passage(other_pool(TRUN, u), L); pre = passage([R.choice(REPL[u])], 20); post = passage([R.choice(REPL[u])], 20)
            if r == 'E-prompt':
                ask('improve this ' + ' '.join(blk), pre + blk + post); paste(blk, lab)
            elif r == 'E-editor':
                type_in(blk, lab); t += R.choice(large_gap); ask('check this ' + ' '.join(blk), pre + blk + post)
            elif r == 'E-editor-auto':  # the environment passes the draft to the chatbot; the prompt does not quote it
                type_in(blk, lab); t += R.choice(large_gap); ask(R.choice(PROM[u]), pre + blk + post)
            else:
                start = len(doc); type_in(blk, 'tmp'); del doc[start:]; del truth[start:]; snap(); t += R.uniform(10, 120)
                ask('fix this ' + ' '.join(blk), pre + blk + post); paste(blk, lab)
        elif r == 'S': paste(list(R.choice(HL)), lab)
        elif r == 'X-ai': paste(passage(other_pool(REPL, u), L), lab)
        else: paste(passage(other_pool(XRUN, u), L), lab)
    # replies of the donor that the simulated student did not use, at random times
    T_end = t
    for ri, rep in enumerate(REPL[u]):
        if ri in used: continue
        tt = R.uniform(0, T_end); turns.append((tt, R.choice(PROM[u]))); replies.append((tt + 10, ' '.join(rep)))
    return dict(final=' '.join(doc), truth=[TRUTH(x) for x in truth], route=list(truth), replies=sorted(replies), turns=sorted(turns), snaps=snaps)

def entry(G, snaps, n):
    """First snapshot containing each n-gram and the tokens inserted in that transition (prefix/suffix trimmed)."""
    need = set(G); te, ins = {}, {}; prev = []
    for t, text in snaps:
        if not need: break
        cur = tokens(text); m = min(len(prev), len(cur)); pre = 0
        while pre < m and prev[pre] == cur[pre]: pre += 1
        suf = 0
        while suf < m - pre and prev[-1 - suf] == cur[-1 - suf]: suf += 1
        a, b = prev[pre:len(prev) - suf], cur[pre:len(cur) - suf]
        if b:
            k = len(b) if (not a or len(b) == 1) else sum(j2 - j1 for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op in ('insert', 'replace'))
            lo, hi = max(0, pre - n + 1), min(len(cur) - n + 1, len(cur) - suf)
            for s in range(lo, max(lo, hi)):
                g = tuple(cur[s:s + n])
                if g in need: te[g] = t; ins[g] = k; need.discard(g)
        prev = cur
    return te, ins

def first_times(items, n):
    f = {}
    for t, text in sorted(items, key=lambda x: x[0]):
        for g in ngrams(tokens(text), n): f.setdefault(g, t)
    return f
SHG = {g for s in SHARED_TXT for g in ngrams(tokens(s), N)}

def attribute(st, snaps, n=N, tau=TAU):
    A = tokens(st['final']); G = ngrams(A, n); tb, tp = first_times(st['replies'], n), first_times(st['turns'], n)
    te, ins = entry(G, snaps, n); lab = {}
    for g in set(G):
        bot = g in tb; pf = bot and g in tp and tp[g] < tb[g]; sh = g in SHG
        if bot and not pf and not sh and g in te and tb[g] <= te[g]: lab[g] = 'C'
        elif bot and (pf or (g in te and te[g] < tb[g])): lab[g] = 'E'
        elif sh: lab[g] = 'S'
        elif g in te: lab[g] = 'T' if ins[g] <= tau else 'X'
        else: lab[g] = 'X'
    rank = {c: i for i, c in enumerate(ORDER)}; best = [9] * len(A)
    for i, g in enumerate(G):
        for j in range(i, i + n): best[j] = min(best[j], rank[lab[g]])
    return A, [ORDER[b] if b < 9 else 'X' for b in best], tb, tp

def thin(snaps, s_):
    if not s_: return snaps
    kept, last = [], -1e18
    for x in snaps:
        if x[0] - last >= s_: kept.append(x); last = x[0]
    if kept[-1] is not snaps[-1]: kept.append(snaps[-1])
    return kept

def online(st):
    att = OnlineAttributor(shared=SHARED_TXT, n=N, tau=TAU); ev = [(t, 0, x) for t, x in st['turns']] + [(t, 1, x) for t, x in st['replies']] + [(t, 2, x) for t, x in st['snaps']]
    lat = []
    for t, k, x in sorted(ev, key=lambda e: (e[0], e[1])):
        if k == 0: att.prompt(t, x)
        elif k == 1: att.reply(t, x)
        else:
            t0 = time.perf_counter(); att.snapshot(t, x); lat.append(time.perf_counter() - t0)
    return att.classify(st['final'])[1], lat

def prf(truth, pred, c):
    tp_ = sum(a == c and b == c for a, b in zip(truth, pred)); pp = sum(b == c for b in pred); ap = sum(a == c for a in truth)
    p = tp_ / pp if pp else None; r = tp_ / ap if ap else None
    return {'precision': p, 'recall': r, 'f1': (2 * p * r / (p + r)) if p and r else None, 'support': ap}
def binprf(truth_c, mark):
    tp_ = sum(a and b for a, b in zip(truth_c, mark)); pp = sum(mark); ap = sum(truth_c)
    p = tp_ / pp if pp else None; r = tp_ / ap if ap else None
    return {'precision': p, 'recall': r, 'f1': (2 * p * r / (p + r)) if p and r else None}
def student_err(true_sh, est_sh):
    a, b = np.array(true_sh), np.array(est_sh)
    return {'mae': float(np.mean(np.abs(b - a))), 'bias': float(np.mean(b - a)), 'spearman': float(stats.spearmanr(a, b)[0]),
            'agree_ge50': float(np.mean((a >= .5) == (b >= .5))), 'true_mean': float(a.mean()), 'est_mean': float(b.mean())}

STUD = [student() for _ in range(NSTUD)]; print('students built', flush=True)
out = {'fit': FIT, 'n_students': NSTUD, 'routes': dict(ROUTES), 'tokens': sum(len(s['truth']) for s in STUD),
       'true_class_shares': {c: sum(x.count(c) for x in (s['truth'] for s in STUD)) / sum(len(s['truth']) for s in STUD) for c in ORDER}}
CONDS = [('full', 0, 5), ('thin5', 5, 5), ('thin15', 15, 5), ('thin60', 60, 5), ('thin300', 300, 5), ('thin60_tau20', 60, 20), ('thin300_tau20', 300, 20)]
RES = {}; PRED = {}
for name, s_, tau in CONDS:
    tr_all, pr_all, rt_all, Us_true, Us_est, T_true, T_est = [], [], [], [], [], [], []
    for st in STUD:
        A, cls, tb, tp = attribute(st, thin(st['snaps'], s_), tau=tau); assert len(A) == len(st['truth'])
        tr_all += st['truth']; pr_all += cls; rt_all += st['route']
        Us_true.append(st['truth'].count('C') / len(A)); Us_est.append(cls.count('C') / len(A))
        T_true.append(st['truth'].count('T') / len(A)); T_est.append(cls.count('T') / len(A))
        if name == 'full': PRED.setdefault('ours', []).append(cls)
    conf = {a: dict(Counter(b for x, b in zip(tr_all, pr_all) if x == a)) for a in ORDER}
    byroute = {r: Counter(b for x, b in zip(rt_all, pr_all) if x == r) for r, _ in ROUTES}
    RES[name] = {'accuracy': float(np.mean([a == b for a, b in zip(tr_all, pr_all)])), 'per_class': {c: prf(tr_all, pr_all, c) for c in ORDER},
                 'confusion': conf, 'route_to_class': {r: {k: v / max(1, sum(c.values())) for k, v in c.items()} for r, c in byroute.items()},
                 'student_Ustar': student_err(Us_true, Us_est), 'student_T_share': student_err(T_true, T_est)}
    print(name, round(RES[name]['accuracy'], 3), {c: (round(RES[name]['per_class'][c]['precision'] or 0, 3), round(RES[name]['per_class'][c]['recall'] or 0, 3)) for c in ORDER}, RES[name]['student_Ustar'], flush=True)
out['attribution'] = RES

# ---------- baselines at full resolution
BL = {k: {'tc': [], 'mk': [], 'true': [], 'est': [], 'Tt': [], 'Te': []} for k in ('ours_offline', 'online', 'naive_overlap', 'direction_blind', 'dialogue_only', 'winnowing', 'string_tiling')}
LAT = []; tt0 = time.perf_counter(); TIMES = Counter()
for k_, st in enumerate(STUD):
    A = tokens(st['final']); truthC = [x == 'C' for x in st['truth']]; rep = [tokens(x) for _, x in st['replies']]; pro = [tokens(x) for _, x in st['turns']]
    tb, tp = first_times(st['replies'], N), first_times(st['turns'], N); prg = {g for p_ in pro for g in ngrams(p_, N)}
    marks = {}
    marks['ours_offline'] = [c == 'C' for c in PRED['ours'][k_]]
    t0 = time.perf_counter(); oc, lat = online(st); TIMES['online'] += time.perf_counter() - t0; LAT += lat; marks['online'] = [c == 'C' for c in oc]
    marks['naive_overlap'] = overlap_mark(A, rep, N)
    src_db = [[g for g in ngrams(r_, N) if g not in prg] for r_ in rep]
    def mark_from(gs):
        S_ = set(gs); m = [False] * len(A)
        for i, g in enumerate(ngrams(A, N)):
            if g in S_:
                for j in range(i, i + N): m[j] = True
        return m
    marks['direction_blind'] = mark_from(g for r_ in src_db for g in r_)
    marks['dialogue_only'] = mark_from(g for g in tb if not (g in tp and tp[g] < tb[g]))
    t0 = time.perf_counter(); marks['winnowing'] = winnow_mark(A, rep, 8, 4); TIMES['winnowing'] += time.perf_counter() - t0
    t0 = time.perf_counter(); marks['string_tiling'] = tile_mark(A, rep, 8); TIMES['string_tiling'] += time.perf_counter() - t0
    for k, m in marks.items():
        BL[k]['tc'] += truthC; BL[k]['mk'] += m; BL[k]['true'].append(np.mean(truthC)); BL[k]['est'].append(np.mean(m))
        BL[k]['Tt'].append(st['truth'].count('T') / len(A))
        BL[k]['Te'].append((PRED['ours'][k_].count('T') / len(A)) if k == 'ours_offline' else (oc.count('T') / len(A) if k == 'online' else 1 - np.mean(m)))
out['baselines'] = {k: {'token_chatbot_text': binprf(v['tc'], v['mk']), 'student_share': student_err(v['true'], v['est']),
                        'typed_share_vs_estimate': student_err(v['Tt'], v['Te'])} for k, v in BL.items()}
out['baselines_note'] = 'typed_share_vs_estimate: estimated T share for ours_offline and online; 1 minus the chatbot share for the overlap baselines (human contribution ratio)'
out['online_latency_ms_per_snapshot'] = {'median': float(np.median(LAT) * 1e3), 'p99': float(np.percentile(LAT, 99) * 1e3), 'max': float(np.max(LAT) * 1e3), 'n': len(LAT)}
out['baseline_seconds_total'] = dict(TIMES)
for k, v in out['baselines'].items(): print(k, v['token_chatbot_text'], v['student_share'], flush=True)
json.dump(out, open('results/r10_benchmark.json', 'w'), indent=1, default=float)
