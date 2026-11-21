"""Analysis log R20, B-41 to B-45 (benchmark). Written after all earlier results were known. No measure, class
definition, parameter or code of the attribution changes; the students of the four R15 test sets are rebuilt exactly.
B-41 the A3 set with the three A3 routes at weights 0.01, 0.05 and 0.10 each (seed 2400): C precision, recall and F1,
and the share of false chatbot labels that fall on A3 routes.
B-42 structural editing of chatbot text: five routes in which a passage of whole reply sentences is edited before it is
pasted, (del) a span of 3-8 tokens deleted from each sentence with probability 0.6, (reorder) the sentences shuffled,
(merge) sentences of two replies alternated, (open) the first 2-5 tokens of each sentence replaced by the writer's own
words, (all) deletion, reordering and new openings together; each at weight 0.05 added to the 13 specified routes
(seed 2800). Per route: share of the surviving reply tokens labelled C, and the share of the pasted block that is reply
text. Reply sentences are split at . ! ? : and line breaks of the raw reply.
B-43 route mixes far from FLoRA: a light-editing cohort (most chatbot text pasted verbatim) and a typing cohort (half of
all blocks typed), seeds 2810 and 2820; C precision/recall/F1, student-level error of U (mean signed and absolute), and
agreement on "at least half chatbot text", for the attribution, n-gram earliest plausible attribution and the R17
classifiers (not retrained).
B-44 standard-form selective risk at the attribution's own coverage for the methods that reach it (13 routes, 20 routes).
B-45 risk-coverage curve of the R17 classifiers, ordered by max(p, 1 - p) for the decision "chatbot text", at the common
coverage of R17 B-29 and at the attribution's coverage, with the area over [0, c*].
Usage: EMB_CACHE=FILE python -I src/r20_benchmark.py DATA_DIR R8_TOKEN_CLASSES WORK_DIR -> results/r20_benchmark.json"""
import sys, os, json, time, pickle, re, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DATA_DIR, R8, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
s17 = open(os.path.join(HERE, 'r17_benchmark.py')).read()
head = s17[:s17.index("out['B30_rules_vs_epa_abstention'] = {}")]
ev_src = s17[s17.index('# ---------- evidence and the fusion classifier'):s17.index("EV = {name:")]
head = head.replace("XNOEV, ARXIV, WORK = sys.argv[3], sys.argv[4], sys.argv[5]", "XNOEV = ARXIV = None; WORK = sys.argv[5]")
head = head.replace("SETS = build([s[0] for s in SPEC]); TEST = ['base', 'ext', 'hard_x', 'a3']", "SETS = build(['base', 'ext', 'hard_x', 'a3']); TEST = ['base', 'ext', 'hard_x', 'a3']")
old_new = "NEW = r15[r15.index('NEW = \"\"\"') + 9: r15.index('\"\"\"\\nanchor')] + '\\n'"
assert old_new in head
NEW2 = """        elif r.startswith('C-struct'):
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri); ask(R.choice(PROM[u]), rep)
            ss = _pick_sents(R, u, ri, L); kind = r[len('C-struct-'):]
            if kind == 'merge':
                rj = R.randrange(len(REPL[u])); used.add(rj); ask(R.choice(PROM[u]), REPL[u][rj]); ss2 = _pick_sents(R, u, rj, L); mix = []
                for q in range(max(len(ss), len(ss2))):
                    if q < len(ss): mix.append(ss[q])
                    if q < len(ss2): mix.append(ss2[q])
                ss = mix
            blk, lb = _struct_edit(R, ss, kind, passage(other_pool(TRUN, u), 400))
            doc.extend(blk); truth.extend([lab if x else 'Tedit' for x in lb]); snap()
"""
head = head.replace(old_new, old_new + " + NEW2")
sys.argv = [os.path.join(HERE, 'r17_benchmark.py'), DATA_DIR, R8, '', '', WORK]
G = {'__name__': 'r17_part', '__file__': os.path.join(HERE, 'r17_benchmark.py'), 'NEW2': NEW2}
exec(compile(head, 'r17_benchmark.py', 'exec'), G)
G_ = G['G_']; log, SETS, tokens, N = G['log'], G['SETS'], G['tokens'], G['N']
attribute, counts, prf_from, ORDER, BASE, EXT, make_student = (G_[k] for k in ('attribute', 'counts', 'prf_from', 'ORDER', 'BASE', 'EXT', 'make_student'))
NS = G['NS']; RCS = dict(G['RCS']); rc_standard = G['rc_standard']; B29 = G['out']['B29_standard_selective']
out = {'n_per_set': NS, 'B29_check': {k: v['ours_token']['risk_at_common_coverage'] for k, v in B29.items()}}

# ---------- reply sentences for the structural-editing routes
SPLIT = re.compile(r'(?<=[.!?:])\s+|\n+')
SENT = {}
for u_, g_ in G_['d'].groupby('user_id'):
    SENT[u_] = [[tokens(s) for s in SPLIT.split(str(x)) if tokens(s)] for x in g_.chatbot_utterance_text.astype(str)]
def _pick_sents(R, u, ri, L):
    ss = SENT[u][ri]
    if len(ss) <= 2: return [list(s) for s in ss]
    s0 = R.randrange(len(ss)); got = []; n_ = 0; q = s0
    while q < len(ss) and (n_ < L or len(got) < 2): got.append(list(ss[q])); n_ += len(ss[q]); q += 1
    return got
def _struct_edit(R, ss, kind, own):
    ss = [list(s) for s in ss]; lab = [[True] * len(s) for s in ss]; oi = [0]
    def ownw(k):
        w = [own[(oi[0] + j) % len(own)] for j in range(k)]; oi[0] += k; return w
    if kind in ('del', 'all'):
        for q, s in enumerate(ss):
            if len(s) >= 10 and R.random() < .6:
                k = R.randint(3, 8); a = R.randrange(1, len(s) - k); del s[a:a + k]; del lab[q][a:a + k]
    if kind in ('reorder', 'all') and len(ss) > 1:
        perm = list(range(len(ss)))
        while perm == sorted(perm): R.shuffle(perm)
        ss = [ss[p] for p in perm]; lab = [lab[p] for p in perm]
    if kind in ('open', 'all'):
        for q, s in enumerate(ss):
            k = min(R.randint(2, 5), max(len(s) - 1, 0))
            if k: ss[q] = ownw(k) + s[k:]; lab[q] = [False] * k + lab[q][k:]
    blk = [x for s in ss for x in s]; lb = [x for l in lab for x in l]; return blk, lb
G_['_pick_sents'] = _pick_sents; G_['_struct_edit'] = _struct_edit
G_['TRUTHMAP']['Tedit'] = 'T'

def build(routes, seed):
    R = random.Random(seed); return [make_student(R, routes) for _ in range(NS)]
def rules_eval(STUD, routes_of_interest=()):
    cn, Ut, Ue = [], [], []; byr = {}; fpA3 = fp = 0
    for st in STUD:
        cls = attribute(st, st['snaps'], entry_rule='token')['cls']; cn.append(counts(st['truth'], cls))
        Ut.append(st['truth'].count('C') / len(cls)); Ue.append(cls.count('C') / len(cls))
        for r, c, t in zip(st['route'], cls, st['truth']):
            byr.setdefault(r, {}); byr[r][c] = byr[r].get(c, 0) + 1
            if c == 'C' and t != 'C': fp += 1; fpA3 += r.startswith('A3')
    o = {c: dict(zip(('precision', 'recall', 'f1'), prf_from(cn, c))) for c in ORDER}
    Ut, Ue = np.array(Ut), np.array(Ue)
    o['U_error'] = {'mean_signed': float((Ue - Ut).mean()), 'mae': float(np.abs(Ue - Ut).mean()), 'true_mean_C': float(Ut.mean()),
                    'agreement_at_half': float(np.mean((Ut >= .5) == (Ue >= .5))), 'true_at_half': float(np.mean(Ut >= .5)), 'est_at_half': float(np.mean(Ue >= .5))}
    o['false_C'] = fp; o['false_C_on_A3_routes'] = fpA3
    o['route_to_class'] = {r: {c: v / sum(d_.values()) for c, v in d_.items()} for r, d_ in byr.items() if (not routes_of_interest or r in routes_of_interest)}
    return o

# ---------- B-41 A3 weight
A3R = ['A3-after', 'A3-before', 'A3-regen']
out['B41_a3_weight'] = {}
for w in (.01, .05, .10):
    S_ = build(BASE + [(r, w) for r in A3R], 2400); o = rules_eval(S_, A3R)
    toks = sum(len(st['truth']) for st in S_); a3t = sum(r.startswith('A3') for st in S_ for r in st['route'])
    o['A3_token_share'] = a3t / toks; out['B41_a3_weight'][str(w)] = o
    log('B41', w, round(o['C']['precision'], 3), round(o['C']['recall'], 3), round(o['A3_token_share'], 3))

# ---------- B-42, B-43 new sets
STRUCT = ['C-struct-del', 'C-struct-reorder', 'C-struct-merge', 'C-struct-open', 'C-struct-all']
LIGHT = [('T', .12), ('M', .03), ('C-paste', .40), ('C-retype', .03), ('C-para10', .03), ('C-para25', .02), ('E-prompt', .05),
         ('E-editor', .03), ('E-editor-auto', .02), ('E-deleted', .02), ('S', .03), ('X-ai', .06), ('X-other', .16)]
TYPED = [('T', .45), ('M', .08), ('C-paste', .05), ('C-retype', .05), ('C-para10', .02), ('C-para25', .01), ('E-prompt', .03),
         ('E-editor', .06), ('E-editor-auto', .06), ('E-deleted', .02), ('S', .03), ('X-ai', .04), ('X-other', .10)]
NEWSETS = {'struct': (BASE + [(r, .05) for r in STRUCT], 2800), 'light': (LIGHT, 2810), 'typed': (TYPED, 2820)}
NS_ = {k: build(v[0], v[1]) for k, v in NEWSETS.items()}; log('new sets built')
exec(compile(ev_src, 'r17_benchmark.py', 'exec'), G)
R17M = pickle.load(open(os.path.join(HERE, '..', 'results', 'r17_fusion_tuned.pkl'), 'rb'))
MODELS = {'classifier_13_routes': R17M['gbt_tuned_train_base'], 'classifier_26_routes': R17M['gbt_tuned_train_all']}
def method_eval(STUD):
    G_['selective'](STUD); rc = G_['_RC']; o = {}
    for m, key in (('attribution', 'ours_token'), ('earliest_plausible', 'earliest_plausible')):
        y = np.array([x[0] for x in rc[key]]); p = np.array([x[1] for x in rc[key]]); tp = (p & y).sum()
        o[m] = {'precision': float(tp / max(p.sum(), 1)), 'recall': float(tp / y.sum()), 'f1': float(2 * tp / (p.sum() + y.sum()))}
    ev = [G['evidence'](st) for st in STUD]; G['ENC'].save()
    Xs = np.vstack([e['X'] for e in ev]); y = np.concatenate([e['y'] for e in ev]) == 0; L = [len(e['y']) for e in ev]; cut_ = np.cumsum(L)[:-1]
    Ut = np.array([yy.mean() for yy in np.split(y, cut_)])
    for mk, mdl in MODELS.items():
        p = mdl.predict(Xs) == 0; tp = (p & y).sum(); Ue = np.array([pp.mean() for pp in np.split(p, cut_)])
        o[mk] = {'precision': float(tp / max(p.sum(), 1)), 'recall': float(tp / y.sum()), 'f1': float(2 * tp / (p.sum() + y.sum())),
                 'U_mean_signed': float((Ue - Ut).mean()), 'U_mae': float(np.abs(Ue - Ut).mean()), 'agreement_at_half': float(np.mean((Ut >= .5) == (Ue >= .5)))}
    return o
out['B42_B43_new_sets'] = {}
for name, STUD in NS_.items():
    o = {'rules': rules_eval(STUD, STRUCT + ['C-para10', 'C-para25', 'C-paste'] if name == 'struct' else ())}
    o['methods'] = method_eval(STUD); out['B42_B43_new_sets'][name] = o
    log(name, {c: round(o['rules'][c]['precision'], 3) for c in 'CT'}, round(o['rules']['C']['recall'], 3), o['rules']['U_error'], {k: round(v['f1'], 3) for k, v in o['methods'].items()})
# share of each structural block that is reply text: count reply tokens and inserted own words per block route
for name in ('struct',):
    tot = {r: [0, 0] for r in STRUCT}
    for st in NS_[name]:
        prev = None
        for rr in st['route']:
            if rr in tot: tot[rr][0] += 1; prev = rr
            elif rr == 'Tedit' and prev in tot: tot[prev][1] += 1
            else: prev = None
    out['B42_B43_new_sets'][name]['reply_share_of_block'] = {r: v[0] / max(sum(v), 1) for r, v in tot.items()}

# ---------- B-44 risk at the attribution's own coverage; B-45 classifier risk-coverage
out['B44_risk_at_attribution_coverage'] = {}; out['B45_classifier_risk_coverage'] = {}
for name in ('base', 'ext'):
    rc = RCS[name]; ca = rc_standard(rc['ours_token'])['coverage']; c_star = B29[name]['common_coverage']
    out['B44_risk_at_attribution_coverage'][name] = {'coverage': ca, **{k: rc_standard(rc[k], ca)['risk_at_common_coverage'] for k in rc if rc_standard(rc[k])['coverage'] >= ca - 1e-9}}
    ev = [G['evidence'](st) for st in SETS[name]]; G['ENC'].save()
    Xs = np.vstack([e['X'] for e in ev]); y = np.concatenate([e['y'] for e in ev]) == 0; o = {}
    for mk, mdl in MODELS.items():
        pr = mdl.predict_proba(Xs)[:, 0]; rows = [(bool(a), bool(b >= .5), float(max(b, 1 - b))) for a, b in zip(y, pr)]
        # predict() takes the arg max over five classes; the binary decision here thresholds p(C) at 0.5
        o[mk] = {'at_c_star': rc_standard(rows, c_star), 'at_attribution_coverage': rc_standard(rows, ca), 'c_star': c_star}
        o[mk]['full_coverage_error'] = float(np.mean([a != b for a, b, _ in rows]))
    out['B45_classifier_risk_coverage'][name] = o
    log('B44/45', name, out['B44_risk_at_attribution_coverage'][name], {k: (round(v['at_c_star']['risk_at_common_coverage'], 4), round(v['at_c_star']['aurc_common_range'], 4)) for k, v in o.items()})
out['seconds'] = time.perf_counter() - G['t00']
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r20_benchmark.json'), 'w'), indent=1, default=float)
log('done')
