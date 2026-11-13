"""Analysis log R12, B-10 to B-12: selective prediction with a stated confidence for every baseline, abstention by
entry size attached to the baselines, error when every token is decided, and the online version with delta > 0.

The simulated students are those of log entry R11 (seed 2100, 13 routes; seed 2200, 20 routes). The data readers,
the student generator and the attribution are taken unchanged from src/r11_benchmark.py: its definitions (everything
before the line "out = {'n_per_seed': 400}") are executed here, so the two scripts cannot drift apart.
Usage: python -I src/r12_benchmark.py DATA_DIR R8_TOKEN_CLASSES -> results/r12_benchmark.json"""
import sys, os, json, random, time
from collections import Counter
from difflib import SequenceMatcher
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'r11_benchmark.py')).read()
# The R11 benchmark drew writers' own and unresolved runs from the token-level FLoRA labels released with log entry R8
# (results/flora_token_classes.csv.gz before R11 re-exported it). The same file is needed to rebuild the same students;
# its path is given as the second argument (git show af6bdb9:results/flora_token_classes.csv.gz > FILE).
if len(sys.argv) > 2: src = src.replace("'results/flora_token_classes.csv.gz'", repr(sys.argv[2]))
cut = src.index("out = {'n_per_seed': 400}")
DATA_DIR = sys.argv[1]
sys.argv = [os.path.join(HERE, 'r11_benchmark.py'), DATA_DIR]
G_ = {'__name__': 'r11_defs', '__file__': os.path.join(HERE, 'r11_benchmark.py')}
exec(compile(src[:cut], 'r11_benchmark.py', 'exec'), G_)
for k in ('make_student', 'BASE', 'EXT', 'attribute', 'scan', 'first_times', 'tokens', 'ngrams', 'N', 'TAU', 'SHARED_TXT',
          'SHG', 'OnlineAttributor', 'overlap_mark', 'winnow_mark', 'tile_mark', 'bin_counts', 'bin_prf'):
    globals()[k] = G_[k]
Q = int(os.environ.get('QUICK', 0))
t00 = time.perf_counter()
R = random.Random(2100); STUD = [make_student(R, BASE) for _ in range(40 if Q else 400)]
R = random.Random(2200); STUDX = [make_student(R, EXT) for _ in range(40 if Q else 400)]
print('students built', time.perf_counter() - t00, flush=True)


def token_iota(A, snaps):
    """Entry size of every token of A under the token entry rule (None: not in the last snapshot, i.e. infinity)."""
    te, ins, last, tl, sid, seg = scan(snaps, N, [])
    io = [None] * len(A); sg = [None] * len(A)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, A, last, autojunk=False).get_opcodes():
        if op == 'equal':
            for q in range(i2 - i1): io[i1 + q] = tl[j1 + q]; sg[i1 + q] = seg[j1 + q]
    return io, sg, last, seg, tl


def big(x): return x is None or x > TAU


def rc_summary(rows):
    """rows: (truthC, predC, confidence); confidence -1 marks abstention; returns AURC, risk at coverages, coverage
    and risk before abstention, error when every token is decided."""
    allt = sorted(rows, key=lambda x: -x[2]); n_ = len(allt)
    err = np.cumsum([a != b for a, b, _ in allt]) / np.arange(1, n_ + 1); cov = np.arange(1, n_ + 1) / n_
    dec = [(a, b) for a, b, c in rows if c != -1]
    return {'aurc': float(np.trapezoid(err, cov)), 'risk_full_coverage': float(err[-1]),
            'coverage_before_abstention': len(dec) / n_, 'risk_before_abstention': float(np.mean([a != b for a, b in dec])) if dec else None,
            'risk_at': {str(c): float(err[int(c * n_) - 1]) for c in (.25, .5, .6, .7, .8, .9, 1.0)},
            'curve': [[float(cov[i]), float(err[i])] for i in np.linspace(0, n_ - 1, 50).astype(int)]}


def selective(STUD_):
    rc = {k: [] for k in ('ours_token', 'earliest_plausible', 'earliest_plausible_iota', 'dialogue_only', 'dialogue_only_iota',
                          'naive_overlap', 'naive_overlap_iota', 'direction_blind', 'paste_log', 'ours_no_abstention')}
    full_err = {k: [0, 0] for k in ('winnowing', 'string_tiling')}
    per = []
    for st in STUD_:
        n0 = {k: len(v) for k, v in rc.items()}
        A = tokens(st['final']); truthC = [x == 'C' for x in st['truth']]; G = ngrams(A, N)
        rep = [tokens(x) for _, x in st['replies']]; pro = [tokens(x) for _, x in st['turns']]
        tb, tp = first_times(st['replies'], N), first_times(st['turns'], N); prg = {g for p_ in pro for g in ngrams(p_, N)}
        repg = set(tb)
        ot = attribute(st, st['snaps'], entry_rule='token', keep=True)
        io, sg, last, seg, tl = token_iota(A, st['snaps'])
        # coverage counts of each token by n-grams in a set
        def share(S_):
            cov = [0] * len(A); hit = [0] * len(A)
            for i, g in enumerate(G):
                for j in range(i, i + N): cov[j] += 1; hit[j] += g in S_
            return [h / c if c else 0.0 for h, c in zip(hit, cov)], hit
        _, inrep = share(repg)
        # attribution (as R11) and the same order without abstention (X decided with the lowest evidential confidence 0)
        rc['ours_token'].extend((a, c == 'C', -1 if c == 'X' else ag) for a, c, ag in zip(truthC, ot['cls'], ot['agree']))
        rc['ours_no_abstention'].extend((a, c == 'C', 0 if c == 'X' else ag) for a, c, ag in zip(truthC, ot['cls'], ot['agree']))
        # earliest plausible attribution: author of each n-gram, token chatbot if any covering n-gram is chatbot;
        # confidence = number of covering n-grams that agree with the token's decision
        te, *_ = scan(st['snaps'], N, G); auth = []
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
        for i, a in enumerate(truthC):
            conf = nC[i] if mk[i] else nW[i]
            rc['earliest_plausible'].append((a, mk[i], conf))
            rc['earliest_plausible_iota'].append((a, mk[i], -1 if (not mk[i] and big(io[i]) and inrep[i] == 0) else conf))
        # overlap baselines with |share - 1/2| as confidence, and the same with abstention by entry size
        for k, S_ in (('dialogue_only', {g for g in tb if not (g in tp and tp[g] < tb[g])}), ('naive_overlap', repg)):
            s, _ = share(S_)
            for i, (a, x) in enumerate(zip(truthC, s)):
                rc[k].append((a, x >= .5, abs(x - .5)))
                rc[k + '_iota'].append((a, x >= .5, -1 if (x < .5 and big(io[i]) and inrep[i] == 0) else abs(x - .5)))
        s, _ = share({g for r_ in rep for g in ngrams(r_, N) if g not in prg})
        rc['direction_blind'].extend((a, x >= .5, abs(x - .5)) for a, x in zip(truthC, s))
        # paste-log rule: chatbot if the insertion (> tau) has >= half of its n-grams in earlier replies, typed if <= tau,
        # unknown otherwise (abstention); confidence: typed 0.5, chatbot |share - 1/2|
        segtoks = {}
        for i_, s_ in enumerate(seg):
            if s_ is not None: segtoks.setdefault(s_, []).append(last[i_])
        stime = {k: t for k, (t, _) in enumerate(st['snaps'])}; repl = [(t, set(ngrams(tokens(x), N))) for t, x in st['replies']]; segsh = {}
        for s_, tk in segtoks.items():
            gs = ngrams(tk, N)
            if len(tk) <= TAU or not gs: segsh[s_] = None; continue
            prior = set().union(*[x for t, x in repl if t <= stime[s_[0]]]) if repl else set()
            segsh[s_] = float(np.mean([g in prior for g in gs]))
        for i, a in enumerate(truthC):
            if io[i] is not None and io[i] <= TAU: rc['paste_log'].append((a, False, .5))
            else:
                x = segsh.get(sg[i]) if sg[i] is not None else None
                if x is not None and x >= .5: rc['paste_log'].append((a, True, x - .5))
                else: rc['paste_log'].append((a, False, -1))
        for k, m in (('winnowing', winnow_mark(A, rep, 8, 4)), ('string_tiling', tile_mark(A, rep, 8))):
            full_err[k][0] += sum(a != b for a, b in zip(truthC, m)); full_err[k][1] += len(A)
        per.append({k: (n0[k], len(rc[k])) for k in rc})
    out = {k: rc_summary(v) for k, v in rc.items()}
    # paired bootstrap over simulated students: AURC of the attribution minus AURC of each method
    def aurc(rows):
        allt = sorted(rows, key=lambda x: -x[2]); n_ = len(allt)
        err = np.cumsum([a != b for a, b, _ in allt]) / np.arange(1, n_ + 1); return float(np.trapezoid(err, np.arange(1, n_ + 1) / n_))
    rb = np.random.default_rng(2130); B = 50 if Q else 1000
    for k in ('earliest_plausible', 'earliest_plausible_iota', 'dialogue_only', 'naive_overlap'):
        d_ = []
        for _ in range(B):
            idx = rb.integers(0, len(per), len(per))
            ro = [x for i in idx for x in rc['ours_token'][per[i]['ours_token'][0]:per[i]['ours_token'][1]]]
            rk = [x for i in idx for x in rc[k][per[i][k][0]:per[i][k][1]]]
            d_.append(aurc(ro) - aurc(rk))
        out[k]['aurc_diff_ours_minus_this'] = out['ours_token']['aurc'] - out[k]['aurc']
        out[k]['ci_aurc_diff'] = [float(np.percentile(d_, 2.5)), float(np.percentile(d_, 97.5))]
        print(k, out[k]['aurc_diff_ours_minus_this'], out[k]['ci_aurc_diff'], flush=True)
    for k, (e, n_) in full_err.items(): out[k] = {'aurc': None, 'risk_full_coverage': e / n_, 'note': 'binary mark without a graded score'}
    return out


class OnlineR11(OnlineAttributor):
    """Online version as released with log entry R11: a reply never turns an entered n-gram into C."""
    def reply(self, t, text):
        for g in set(ngrams(tokens(text), self.n)):
            if g in self.t_bot: continue
            self.t_bot[g] = t
            if g in self.entry and self.entry[g][0] + self.delta < t and self.label[g] in "TXS":
                self.label[g] = "E"; self.changed.add(g)


def online_vs_offline(STUD_, delta, cls_=None):
    """Online labels at submission against offline labels with the same delta; transitions of n-gram labels after entry."""
    agree = tot = 0; trans = Counter(); byroute = {}
    for st in STUD_:
        off = attribute(st, st['snaps'], entry_rule='ngram', delta=delta)['cls']
        att = (cls_ or OnlineAttributor)(shared=SHARED_TXT, n=N, tau=TAU, delta=delta); first = {}; tfirst = {}
        ev = [(t, 0, x) for t, x in st['turns']] + [(t, 1, x) for t, x in st['replies']] + [(t, 2, x) for t, x in st['snaps']]
        for t, k, x in sorted(ev, key=lambda e: (e[0], e[1])):
            if k == 0: att.prompt(t, x)
            elif k == 1:
                before = dict(att.label); att.reply(t, x)
                for g, l in att.label.items():
                    if before.get(g) != l: tfirst[g] = (t - att.entry[g][0])
            else:
                for g, l in att.snapshot(t, x).items(): first.setdefault(g, l)
        on = att.classify(st['final'])[1]
        agree += sum(a == b for a, b in zip(off, on)); tot += len(on)
        for g, l in first.items():
            if l != att.label[g]:
                lag = tfirst.get(g, np.nan)
                trans[f'{l}->{att.label[g]}' + ('_within_delta' if lag <= delta else '_after_delta')] += 1
        for r_, a, b in zip(st['route'], off, on):
            byroute.setdefault(r_, [0, 0]); byroute[r_][0] += a == b; byroute[r_][1] += 1
    return {'token_agreement': agree / tot, 'n_tokens': tot, 'transitions': dict(trans),
            'agreement_by_route': {r_: a / n_ for r_, (a, n_) in byroute.items()}}


out = {}
out['B10_selective_base'] = selective(STUD); print('B10 base', {k: (v['aurc'], v['risk_full_coverage']) for k, v in out['B10_selective_base'].items()}, flush=True)
out['B10_selective_ext'] = selective(STUDX); print('B10 ext', {k: (v['aurc'], v['risk_full_coverage']) for k, v in out['B10_selective_ext'].items()}, flush=True)
out['B11_online_delta0_base'] = online_vs_offline(STUD, 0.0)
out['B11_online_delta0_ext'] = online_vs_offline(STUDX, 0.0)
out['B11_online_delta30_ext'] = online_vs_offline(STUDX, 30.0)
out['B11_online_delta30_ext_R11_version'] = online_vs_offline(STUDX, 30.0, OnlineR11)
for k in ('B11_online_delta0_base', 'B11_online_delta0_ext', 'B11_online_delta30_ext', 'B11_online_delta30_ext_R11_version'): print(k, out[k]['token_agreement'], out[k]['transitions'], flush=True)
# B-12: check that the students are those of R11 (attribution F1 for chatbot text, seed 2100)
bc = [bin_counts([x == 'C' for x in st['truth']], [c == 'C' for c in attribute(st, st['snaps'], entry_rule='token')['cls']]) for st in STUD]
out['B12_check_same_students_f1'] = bin_prf(bc)[2]
out['seconds'] = time.perf_counter() - t00
json.dump(out, open('results/r12_benchmark_quick.json' if Q else 'results/r12_benchmark.json', 'w'), indent=1, default=float)
print('done', out['seconds'], out['B12_check_same_students_f1'])
