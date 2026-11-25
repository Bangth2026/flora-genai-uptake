"""Analysis log R29 (FLoRA), written after a review of the fourteenth version of the manuscript, after the analyses ran.
Nothing in the attribution changes.
STEP audits: src/r17_flora.py executed unchanged on the whole system trace (DATA_FULL; entry R28), except for its output
  paths: F-23 a new keystroke audit of T and X passages drawn from all writers with keystrokes (seed 2171, at most three
  passages per writer, 50 per stratum), F-24 the census of echo passages, and the other R17 outputs; then F-67 a census of
  every T and X run of at least eight tokens with the criteria of the R9 audit (T agrees if keystrokes cover at least half
  of the characters, X if less than a fifth), with student-bootstrap intervals (2,000 resamples, seed 2901).
STEP pastes: F-69 src/r11_flora.py executed unchanged on the whole trace except for its output paths (labels written to a
  scratch file and compared with the released labels; results/r29_r11_flora.json): logged pastes by dominant class (Table B28).
STEP lower: F-68 the lower end of Table 4 (C tokens that entered in insertions of more than tau tokens) with tau = 2.
Usage: python -I src/r29_flora.py DATA_FULL_DIR STEP -> results/r29_flora_STEP.json (audits also results/r29_audit_sample.csv)"""
import sys, os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
D, STEP = sys.argv[1], sys.argv[2]
if STEP == 'audits':
    src = open(os.path.join(HERE, 'r17_flora.py')).read()
    src = src.replace("os.path.join(HERE, '..', 'results', 'r17_flora.json')", "os.path.join(HERE, '..', 'results', 'r29_flora_audits.json')")
    src = src.replace("os.path.join(HERE, '..', 'results', 'r17_audit_sample.csv')", "os.path.join(HERE, '..', 'results', 'r29_audit_sample.csv')")
    assert 'r29_flora_audits.json' in src and 'r29_audit_sample.csv' in src
    # an echo run with no covering 8-gram in a reply (editor route) or in a prompt (prompt route) cannot be scored (none
    # occurred among the 71 students read by R17); such runs are skipped, and their number is recorded
    g0 = "            tb = min(t['tbot'][g] for g in pg if g in t['tbot']); ev ="
    assert src.count(g0) == 1
    src = src.replace(g0, "            if not any(g in t['tbot'] for g in pg): SKIPPED.append(u); continue\n" + g0)
    g1 = "            tp = np.datetime64(min(t['tpr'][g] for g in pg if g in t['tpr'])); w0 ="
    assert src.count(g1) == 1
    src = src.replace(g1, "            if not any(g in t['tpr'] for g in pg): SKIPPED.append(u); continue\n" + g1)
    sys.argv = [os.path.join(HERE, 'r17_flora.py'), D]
    G = {'__name__': '__main__', '__file__': os.path.join(HERE, 'r17_flora.py'), 'SKIPPED': []}
    exec(compile(src, 'r17_flora.py', 'exec'), G)
    TK, KEYU, runs_of, keys_verdict, boot_share = (G[k] for k in ('TK', 'KEYU', 'runs_of', 'keys_verdict', 'boot_share'))
    rng = np.random.default_rng(2901); rows = []
    for u in KEYU:
        for c, i, j in runs_of(TK[u]['var'], 'TX'):
            k, ch = keys_verdict(u, i, j)
            rows.append(dict(u=u, cls=c, tokens=j - i, typed=(k is not None and k >= .5 * ch), not_typed=(k is None or k < .2 * ch)))
    R = pd.DataFrame(rows); o = {}
    for c, crit in (('T', 'typed'), ('X', 'not_typed')):
        x = R[R.cls == c]
        o[c] = {'runs': int(len(x)), 'students': int(x.u.nunique()), 'tokens': int(x.tokens.sum()), 'agree': boot_share(x[crit], x.u, rng),
                'agree_token_weighted': float((x[crit] * x.tokens).sum() / x.tokens.sum())}
    out = json.load(open(os.path.join(ROOT, 'results', 'r29_flora_audits.json'))); out['F67_census_T_X'] = o; out['F24_echo_runs_skipped_no_reply_ngram'] = len(G['SKIPPED'])
    json.dump(out, open(os.path.join(ROOT, 'results', 'r29_flora_audits.json'), 'w'), indent=1, default=float)
    print('F67', o, flush=True)
elif STEP == 'pastes':
    src = open(os.path.join(HERE, 'r11_flora.py')).read()
    for a, b in [(".to_csv('results/flora_token_classes.csv.gz', index=False, compression='gzip')", ".to_csv('/tmp/r29_r11_token_classes.csv.gz', index=False, compression='gzip')"),
                 ("sc_.to_csv('results/flora_student_classes.csv'", "sc_.to_csv('/tmp/r29_r11_student_classes.csv'"),
                 ("open('results/r11_flora.json', 'w')", "open('results/r29_r11_flora.json', 'w')")]:
        assert src.count(a) == 1, a; src = src.replace(a, b)
    os.chdir(ROOT); sys.argv = [os.path.join(HERE, 'r11_flora.py'), D]
    G = {'__name__': '__main__', '__file__': os.path.join(HERE, 'r11_flora.py')}
    exec(compile(src, 'r11_flora.py', 'exec'), G)
    a = pd.read_csv('/tmp/r29_r11_token_classes.csv.gz'); b = pd.read_csv('results/flora_token_classes.csv.gz')
    o = json.load(open('results/r29_r11_flora.json')); o['F69_labels_identical_to_released'] = bool(a['class'].equals(b['class']))
    json.dump(o, open('results/r29_r11_flora.json', 'w'), indent=1, default=float); print('labels identical', o['F69_labels_identical_to_released'])
elif STEP == 'lower':
    src23 = open(os.path.join(HERE, 'r23_flora.py')).read()
    cut = src23.index('# ---------- F-51 partial identification')
    G = {'__name__': 'r23_defs', '__file__': os.path.join(HERE, 'r23_flora.py')}
    sys.argv = [os.path.join(HERE, 'r23_flora.py'), D]
    exec(compile(src23[:cut], 'r23_flora.py', 'exec'), G)
    TK, users = G['TK'], G['users']; rng = np.random.default_rng(2902)
    v = {t: np.array([float(np.mean((TK[u]['lab'] == 'C') & (TK[u]['TEV'][t] == 2))) for u in users]) for t in (2, 5)}
    def at(x, q):
        b = [np.mean(x[rng.integers(0, len(x), len(x))] >= q) for _ in range(2000)]
        return [float(np.mean(x >= q)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    o = {f'tau{t}': {'mean': float(x.mean()), **{f'at_or_above_{int(q * 100)}': at(x, q) for q in (.25, .5, .75)}} for t, x in v.items()}
    json.dump({'F68_lower_end_by_tau': o}, open(os.path.join(ROOT, 'results', 'r29_flora_lower.json'), 'w'), indent=1)
    print('F68', o)
