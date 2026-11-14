"""Analysis log R14, B-16 to B-18 (benchmark): the risk-coverage difference with two routes left out together and the
route whose removal widens it; snapshots thinned to 300 s with the current code; the shared-source class across
snapshot resolutions.

The simulated students, the attribution and the scores are those of log entry R13: the definitions of
src/r13_benchmark.py (everything before the line "out = {}") are executed here.
Usage: python -I src/r14_benchmark.py DATA_DIR R8_TOKEN_CLASSES -> results/r14_benchmark.json"""
import sys, os, json, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
src13 = open(os.path.join(HERE, 'r13_benchmark.py')).read()
cut13 = src13.index("\nout = {}\n")
G3 = {'__name__': 'r13_defs', '__file__': os.path.join(HERE, 'r13_benchmark.py')}
t00 = time.perf_counter()
exec(compile(src13[:cut13], 'r13_benchmark.py', 'exec'), G3)
STUD, STUDX, selective, aurc, G2 = G3['STUD'], G3['STUDX'], G3['selective'], G3['aurc'], G3['G2']
G_ = G2['G_']; run_conditions, attribute, thin = G_['run_conditions'], G_['attribute'], G_['thin']


def leave_out(STUD_, groups):
    selective(STUD_); RC = G2['_RC']
    routes = [r_ for st in STUD_ for r_ in st['route']]
    O, E = RC['ours_token'], RC['earliest_plausible_iota']
    out = {'all': aurc(O) - aurc(E)}
    for name, grp in groups.items():
        keep = [i for i, x in enumerate(routes) if x not in grp]
        out[name] = aurc([O[i] for i in keep]) - aurc([E[i] for i in keep])
    return out


out = {}
ext_routes = sorted({r for st in STUDX for r in st['route']})
groups = {'S+C-stream': {'S', 'C-stream'}}
groups.update({f'S+C-stream+{r}': {'S', 'C-stream', r} for r in ext_routes if r not in ('S', 'C-stream')})
groups.update({r: {r} for r in ext_routes})
out['B16_leave_out_ext'] = leave_out(STUDX, groups)
print('B16', {k: round(v, 4) for k, v in sorted(out['B16_leave_out_ext'].items(), key=lambda kv: kv[1])[:5]}, out['B16_leave_out_ext']['S+C-stream'], flush=True)
base_routes = sorted({r for st in STUD for r in st['route']})
out['B16_leave_out_base'] = leave_out(STUD, {r: {r} for r in base_routes})

conds = [(f'thin{s_}_tau{tau}_{rule}', s_, tau, rule, 0) for s_ in (0, 60, 300) for tau in (5, 20) for rule in ('ngram', 'token')]
out['B17_thinning_current_code'] = run_conditions(STUD, conds)

# B-18: where the reading-highlight route goes at each resolution (n-gram labels of S-route tokens)
res18 = {}
for s_ in (0, 5, 15, 60, 300):
    cnt = Counter()
    for st in STUD:
        cls = attribute(st, thin(st['snaps'], s_), entry_rule='token')['cls']
        for r_, c in zip(st['route'], cls):
            if r_ == 'S': cnt[c] += 1
    res18[str(s_)] = dict(cnt)
out['B18_highlight_route_by_resolution'] = res18
print('B18', res18, flush=True)
out['seconds'] = time.perf_counter() - t00
json.dump(out, open('results/r14_benchmark.json', 'w'), indent=1, default=float)
print('done', out['seconds'])
