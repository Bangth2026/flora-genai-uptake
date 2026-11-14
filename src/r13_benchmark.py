"""Analysis log R13, B-13 to B-15 (benchmark): offline and online labels with delta > 0 under the decision list as
stated; where the remaining difference between the attribution and earliest plausible attribution with abstention lies.

The simulated students, the baselines and the selective-prediction code are those of log entry R12: the definitions of
src/r12_benchmark.py (everything before the line "out = {}"), which in turn executes those of src/r11_benchmark.py, are
executed here. Two corrections apply only with delta > 0 and leave every delta = 0 result unchanged:
  - offline: R2 (editor-first echo) requires t_W(g) + delta < t_R(g), as stated; the R11/R12 code tested
    t_W(g) < t_R(g), which differs only for a shared n-gram whose first reply came within delta after its entry;
  - online: a reply within delta of entry turns an S label into E when a writer turn contained the n-gram before
    the reply (R1 precedes R3); the R12 version left S unchanged.
Usage: python -I src/r13_benchmark.py DATA_DIR R8_TOKEN_CLASSES -> results/r13_benchmark.json"""
import sys, os, json, time
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'demo'))
src12 = open(os.path.join(HERE, 'r12_benchmark.py')).read()
cut12 = src12.index("\nout = {}\n")
# keep the per-token rows of the selective-prediction comparison for the decomposition by route
src12 = src12.replace("    out = {k: rc_summary(v) for k, v in rc.items()}", "    globals()['_RC'] = rc\n    out = {k: rc_summary(v) for k, v in rc.items()}")
src12 = src12.replace("    for k in ('earliest_plausible', 'earliest_plausible_iota', 'dialogue_only', 'naive_overlap'):", "    for k in ('earliest_plausible_iota',):")
# R2 with delta, as stated in the article (only differs for delta > 0); applied inside the R11 definitions
src12 = src12.replace("exec(compile(src[:cut], 'r11_benchmark.py', 'exec'), G_)",
                      "src = src.replace(\"elif bot and (pf or (g in te and te[g] < tb[g])): lab[g] = 'E'\", \"elif bot and (pf or (g in te and te[g] + delta < tb[g])): lab[g] = 'E'\")\n"
                      "assert 'te[g] + delta < tb[g]' in src\n"
                      "cut = src.index(\"out = {'n_per_seed': 400}\")\n"
                      "exec(compile(src[:cut], 'r11_benchmark.py', 'exec'), G_)")
G2 = {'__name__': 'r12_defs', '__file__': os.path.join(HERE, 'r12_benchmark.py')}
sys.argv = [os.path.join(HERE, 'r12_benchmark.py')] + sys.argv[1:]
t00 = time.perf_counter()
exec(compile(src12[:cut12], 'r12_benchmark.py', 'exec'), G2)
STUD, STUDX, attribute, selective, OnlineAttributor = G2['STUD'], G2['STUDX'], G2['attribute'], G2['selective'], G2['OnlineAttributor']
tokens, ngrams, N, TAU, SHARED_TXT, SHG = (G2[k] for k in ('tokens', 'ngrams', 'N', 'TAU', 'SHARED_TXT', 'SHG'))
first_times, scan, bin_counts, bin_prf = G2['first_times'], G2['scan'], G2['bin_counts'], G2['bin_prf']
from provenance_online import OnlineAttributor as OnlineR13   # corrected version (demo/)


class OnlineR12(OnlineR13):
    """Online version as released with log entry R12: a reply within delta of entry changes only T or X labels."""
    def reply(self, t, text):
        for g in set(ngrams(tokens(text), self.n)):
            if g in self.t_bot: continue
            self.t_bot[g] = t
            if g not in self.entry: continue
            te = self.entry[g][0]
            if te + self.delta < t:
                if self.label[g] in "TXS": self.label[g] = "E"; self.changed.add(g)
            elif self.label[g] in "TX":
                self.label[g] = "E" if (g in self.t_pr and self.t_pr[g] < t) else "C"; self.changed.add(g)


def offline_R12(st, delta):
    """Offline n-gram rule of R11/R12 (R2 tested as t_W < t_R), for the comparison."""
    A = tokens(st['final']); G = ngrams(A, N); tb, tp = first_times(st['replies'], N), first_times(st['turns'], N)
    te, ins, *_ = scan(st['snaps'], N, G); lab = {}
    for g in set(G):
        bot = g in tb; pf = bot and g in tp and tp[g] < tb[g]; sh = g in SHG
        if bot and not pf and not sh and g in te and tb[g] <= te[g] + delta: lab[g] = 'C'
        elif bot and (pf or (g in te and te[g] < tb[g])): lab[g] = 'E'
        elif sh: lab[g] = 'S'
        elif g in te: lab[g] = 'T' if ins[g] <= TAU else 'X'
        else: lab[g] = 'X'
    return lab


def run_online(st, delta, cls_):
    att = cls_(shared=SHARED_TXT, n=N, tau=TAU, delta=delta)
    ev = [(t, 0, x) for t, x in st['turns']] + [(t, 1, x) for t, x in st['replies']] + [(t, 2, x) for t, x in st['snaps']]
    for t, k, x in sorted(ev, key=lambda e: (e[0], e[1])):
        if k == 0: att.prompt(t, x)
        elif k == 1: att.reply(t, x)
        else: att.snapshot(t, x)
    return att


def compare(STUD_, delta):
    """Token agreement of online and offline labels for the four combinations of versions; diagnosis of every
    n-gram whose online and offline labels differ."""
    res = {}
    for tag_on, cls_ in (('online_R12', OnlineR12), ('online_R13', OnlineR13)):
        for tag_off in ('offline_R12', 'offline_R13'):
            res[f'{tag_on}_vs_{tag_off}'] = {'agree': 0, 'tot': 0, 'by_route_disagree': Counter(), 'ngram_cases': Counter()}
    for st in STUD_:
        A = tokens(st['final']); G = ngrams(A, N)
        offs = {'offline_R13': attribute(st, st['snaps'], entry_rule='ngram', delta=delta, keep=True)}
        lab13 = offs['offline_R13']['lab']; lab12 = offline_R12(st, delta)
        rank = {c: i for i, c in enumerate('CESTX')}
        def proj(lab):
            best = [9] * len(A)
            for i, g in enumerate(G):
                for j in range(i, i + N): best[j] = min(best[j], rank[lab[g]])
            return ['CESTX'[b] if b < 9 else 'X' for b in best]
        cls_off = {'offline_R13': offs['offline_R13']['cls'], 'offline_R12': proj(lab12)}
        labs_off = {'offline_R13': lab13, 'offline_R12': lab12}
        tb, tp = first_times(st['replies'], N), first_times(st['turns'], N)
        for tag_on, cls_ in (('online_R12', OnlineR12), ('online_R13', OnlineR13)):
            att = run_online(st, delta, cls_); on = att.classify(st['final'])[1]
            for tag_off in ('offline_R12', 'offline_R13'):
                r = res[f'{tag_on}_vs_{tag_off}']; off = cls_off[tag_off]
                r['agree'] += sum(a == b for a, b in zip(off, on)); r['tot'] += len(on)
                for rt, a, b in zip(st['route'], off, on):
                    if a != b: r['by_route_disagree'][rt] += 1
                for g in set(G):
                    lo, ln = labs_off[tag_off][g], att.label.get(g, 'X')
                    if lo != ln:
                        te_ = att.entry[g][0] if g in att.entry else None
                        pf = g in tp and g in tb and tp[g] < tb[g]
                        lag = None if (te_ is None or g not in tb) else tb[g] - te_
                        case = (f'offline {lo} / online {ln}; shared={g in SHG}; turn before reply={pf}; '
                                f'reply after entry within delta={lag is not None and 0 < lag <= delta}')
                        r['ngram_cases'][case] += 1
    for k, r in res.items():
        r['token_agreement'] = r['agree'] / r['tot']; r['tokens_disagree'] = r['tot'] - r['agree']
        r['by_route_disagree'] = dict(r['by_route_disagree']); r['ngram_cases'] = dict(r['ngram_cases'])
    return res


def route_shares(STUD_, delta):
    """Share of each route's tokens in each class (offline, n-gram rule as in R11 Table 4 and token rule)."""
    out = {}
    for rule in ('token', 'ngram'):
        cnt = {}
        for st in STUD_:
            cls = attribute(st, st['snaps'], entry_rule=rule, delta=delta)['cls']
            for rt, c in zip(st['route'], cls): cnt.setdefault(rt, Counter())[c] += 1
        out[rule] = {rt: {c: v / sum(x.values()) for c, v in x.items()} for rt, x in cnt.items()}
    bc = [bin_counts([x == 'C' for x in st['truth']], [c == 'C' for c in attribute(st, st['snaps'], entry_rule='token', delta=delta)['cls']]) for st in STUD_]
    p, r, f = bin_prf(bc); out['C_precision_recall_f1_token'] = [p, r, f]
    return out


def aurc(rows):
    allt = sorted(rows, key=lambda x: -x[2]); n_ = len(allt)
    err = np.cumsum([a != b for a, b, _ in allt]) / np.arange(1, n_ + 1)
    return float(np.trapezoid(err, np.arange(1, n_ + 1) / n_))


def decompose(STUD_):
    """Where the AURC difference between the attribution and earliest plausible attribution with abstention by entry
    size comes from: per route, the tokens each method decides and gets wrong, and the AURC difference when the
    route's tokens are left out."""
    selective(STUD_); RC = G2['_RC']
    routes = [r_ for st in STUD_ for r_ in st['route']]
    O, E = RC['ours_token'], RC['earliest_plausible_iota']
    assert len(O) == len(E) == len(routes)
    full = aurc(O) - aurc(E); per = {}
    for rt in sorted(set(routes)):
        idx = [i for i, x in enumerate(routes) if x == rt]
        o_dec = [O[i] for i in idx if O[i][2] != -1]; e_dec = [E[i] for i in idx if E[i][2] != -1]
        keep = [i for i, x in enumerate(routes) if x != rt]
        per[rt] = {'tokens': len(idx),
                   'ours_decided': len(o_dec), 'ours_wrong_decided': sum(a != b for a, b, _ in o_dec),
                   'epa_iota_decided': len(e_dec), 'epa_iota_wrong_decided': sum(a != b for a, b, _ in e_dec),
                   'ours_abstains_epa_iota_wrong': sum(O[i][2] == -1 and E[i][2] != -1 and E[i][0] != E[i][1] for i in idx),
                   'epa_iota_abstains_ours_wrong': sum(E[i][2] == -1 and O[i][2] != -1 and O[i][0] != O[i][1] for i in idx),
                   'aurc_diff_without_route': aurc([O[i] for i in keep]) - aurc([E[i] for i in keep])}
    return {'aurc_diff_all': full, 'by_route': per}


out = {}
out['B13_online_vs_offline_delta30_ext'] = compare(STUDX, 30.0)
out['B13_online_vs_offline_delta30_base'] = compare(STUD, 30.0)
out['B13_online_vs_offline_delta0_ext'] = compare(STUDX, 0.0)
for k in ('B13_online_vs_offline_delta30_ext', 'B13_online_vs_offline_delta30_base', 'B13_online_vs_offline_delta0_ext'):
    print(k, {a: (b['token_agreement'], b['tokens_disagree'], b['ngram_cases']) for a, b in out[k].items()}, flush=True)
out['B14_route_shares_delta30_ext_offline_R13'] = route_shares(STUDX, 30.0)
print('B14', out['B14_route_shares_delta30_ext_offline_R13']['C_precision_recall_f1_token'], flush=True)
out['B15_aurc_decomposition_ext'] = decompose(STUDX)
out['B15_aurc_decomposition_base'] = decompose(STUD)
for k in ('B15_aurc_decomposition_ext', 'B15_aurc_decomposition_base'):
    print(k, out[k]['aurc_diff_all'], flush=True)
    for rt, v in sorted(out[k]['by_route'].items(), key=lambda kv: kv[1]['aurc_diff_without_route']):
        print('  ', rt, v, flush=True)
out['seconds'] = time.perf_counter() - t00
json.dump(out, open('results/r13_benchmark_quick.json' if G2['Q'] else 'results/r13_benchmark.json', 'w'), indent=1, default=float)
print('done', out['seconds'])
