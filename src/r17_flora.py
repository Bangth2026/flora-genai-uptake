"""Analysis log R17, F-22 to F-27 (FLoRA).
F-22 the echo class split by the route that made it echo (writer turn first, R1; editor first, R2) and by entry size
(at most tau tokens, or more), with the share of students at or above half under U and under U with the large-entry
echo added, as bounds; F-23 a new keystroke audit of T and X passages, drawn from the 70 keystroke students with a new
seed and excluding every passage of the R9 and R9b samples, scored for both entry rules under the criteria fixed in R9;
F-24 keystroke support of every echo passage of the 70 students, by route and entry size (a census, not a sample);
F-25 how much of the large-entry echo can be traced to a logged paste that also carried text of the student's own
earlier replies, or to replies given to other students; F-26 the peer threshold of the shared-source rule as an
absolute count and as a share of the cohort; F-27 chance coverage by the replies of other students for n = 5 to 12.
The labels are those of the attribution (src/r9b_flora_token_entry.py executed up to its audit); nothing changes in them.
Written before these analyses ran; see ANALYSIS_LOG_R17.md.
Usage: python -I src/r17_flora.py DATA_DIR -> results/r17_flora.json"""
import sys, os, json, time, re
from collections import Counter
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1]
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)
src = open(os.path.join(HERE, 'r9b_flora_token_entry.py')).read()
src = src.replace("    TK[u] = dict(F=F, lab=lab, var=var, tt=tt, times=times)",
                  "    TK[u] = dict(F=F, lab=lab, var=var, tt=tt, times=times, te=te, cls=cls, FG=FG, FGs=FGs, ted=sc['ted'], tbot=tbot, tpr=tpr)")
cut = src.index('# keystroke support per labelled run')
sys.argv = [os.path.join(HERE, 'r9b_flora_token_entry.py'), D]
G = {'__name__': 'r9b_defs', '__file__': os.path.join(HERE, 'r9b_flora_token_entry.py')}
exec(compile(src[:cut], 'r9b_flora_token_entry.py', 'exec'), G)
TK, users, KEYU, EKT, grams, tok, d, H, Cc, P = (G[k] for k in ('TK', 'users', 'KEYU', 'EKT', 'grams', 'tok', 'd', 'H', 'Cc', 'P'))
NB = 2000; N = 8; TAU = 5
log('labels', len(users))
out = {}

# ---------- trace: chat keystrokes, paste shortcuts, chatbot-panel selections, logged pastes (as in src/r9_flora.py)
tr = pd.read_pickle(f'{D}/trace.pkl'); tr['t'] = pd.to_datetime(tr.log_trace_time, unit='ms')
ek = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy()
ck = tr[(tr.learning_action_label == 'ASK_MULTI_AGENTS') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy()
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); del tr
pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
def keyinfo(df):
    df = df.sort_values(['user_id', 't']).copy(); v = df.event_value.astype(str).str.replace('KEY:::', '', regex=False)
    key = v.str.split('---').str[0]
    df['printable'] = key.str.len().eq(1).values; df['mod'] = key.isin(['Control', 'Meta']).values; df['isv'] = v.str.endswith('KeyV').values
    return df
ek, ck = keyinfo(ek), keyinfo(ck)
def pastes_shortcut(df):
    o = {}
    for u, x in df.groupby('user_id'):
        res = []; lastmod = None
        for t_, m_, v_ in zip(x.t.values, x['mod'].values, x.isv.values):
            if m_: lastmod = t_
            elif v_ and lastmod is not None and (t_ - lastmod) / np.timedelta64(1, 's') <= 1.5: res.append(t_)
        o[u] = np.array(res, dtype='datetime64[ns]')
    return o
EPS, CPS = pastes_shortcut(ek), pastes_shortcut(ck)
CKT = {u: x.t.values[x.printable.values] for u, x in ck.groupby('user_id')}
PASTEU = sorted(set(pa.user_id) & set(users))
PG = {u: [set(grams(tok(s), N)) for s in x.txt] for u, x in pa.groupby('user_id')}
log('trace')

def boot_share(vals, groups, rng, nb=NB):
    """mean of a 0/1 value with a 95% interval from resampling students"""
    vals = np.asarray(vals, float); groups = np.asarray(groups); us = np.unique(groups); idx = {q: np.flatnonzero(groups == q) for q in us}; bs = []
    for _ in range(nb):
        b = rng.choice(us, len(us)); ii = np.concatenate([idx[q] for q in b]); bs.append(vals[ii].mean())
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def runs_of(L, labels, minlen=8):
    r = []; i = 0
    while i < len(L):
        j = i
        while j < len(L) and L[j] == L[i]: j += 1
        if j - i >= minlen and L[i] in labels: r.append((L[i], i, j))
        i = j
    return r

# ---------- F-22 the echo class by route and entry size, and the bounds on the share at or above half
def e_route(u):
    """per token: 'prompt' if a covering echo 8-gram reached the writer's turn before the reply (R1), 'editor' otherwise"""
    t = TK[u]; F = t['F']; r = np.array([''] * len(F), dtype=object)
    for i, g in enumerate(t['FG']):
        if t['cls'][g] != 'E': continue
        pr = g in t['tpr'] and g in t['tbot'] and t['tpr'][g] < t['tbot'][g]
        for j in range(i, i + N):
            if t['lab'][j] == 'E' and (pr or r[j] == ''): r[j] = 'prompt' if pr else (r[j] or 'editor')
    return r
ER = {u: e_route(u) for u in users}
share = lambda m: float(m.mean()) if len(m) else 0.0
rows = []
for u in users:
    t = TK[u]; lab = t['lab']; big = t['te'] == 2; E = lab == 'E'
    rows.append(dict(u=u, C=share(lab == 'C'), E=share(E), E_prompt=share(E & (ER[u] == 'prompt')), E_editor=share(E & (ER[u] == 'editor')),
                     E_small=share(E & ~big), E_large=share(E & big), E_prompt_large=share(E & big & (ER[u] == 'prompt')),
                     E_editor_large=share(E & big & (ER[u] == 'editor'))))
S = pd.DataFrame(rows).set_index('u')
rng = np.random.default_rng(2171)
def at_half(x):
    v = (x >= .5).astype(float).values; bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
tokE = sum(int((TK[u]['lab'] == 'E').sum()) for u in users)
pool = lambda m: sum(int(m(u).sum()) for u in users) / tokE
out['F22_echo_decomposition'] = {
    'mean_share_of_proposal': {k: float(S[k].mean()) for k in S.columns},
    'pooled_share_of_E_tokens': {'prompt_first': pool(lambda u: (TK[u]['lab'] == 'E') & (ER[u] == 'prompt')),
                                 'editor_first': pool(lambda u: (TK[u]['lab'] == 'E') & (ER[u] == 'editor')),
                                 'large_entry': pool(lambda u: (TK[u]['lab'] == 'E') & (TK[u]['te'] == 2)),
                                 'prompt_first_large': pool(lambda u: (TK[u]['lab'] == 'E') & (TK[u]['te'] == 2) & (ER[u] == 'prompt')),
                                 'editor_first_large': pool(lambda u: (TK[u]['lab'] == 'E') & (TK[u]['te'] == 2) & (ER[u] == 'editor'))},
    'students_at_or_above_half': {'U': at_half(S.C), 'U_plus_E_large': at_half(S.C + S.E_large),
                                  'U_plus_E_prompt_large': at_half(S.C + S.E_prompt_large), 'U_plus_E_editor_large': at_half(S.C + S.E_editor_large),
                                  'U_plus_E_all': at_half(S.C + S.E)},
    'mean_share_bounds': {'U': float(S.C.mean()), 'U_plus_E_large': float((S.C + S.E_large).mean()), 'U_plus_E_all': float((S.C + S.E).mean())}}
log('F22', out['F22_echo_decomposition']['students_at_or_above_half'])

# ---------- F-23 new keystroke audit of T and X, both entry rules, passages not in the R9 or R9b samples
def keys_verdict(u, i, j):
    t = TK[u]; ks = t['tt'][i:j]; ks = ks[ks >= 0]
    chars = len(' '.join(t['F'][i:j]))
    if not len(ks): return None, chars
    t1 = np.datetime64(t['times'][ks.min()]) - np.timedelta64(3, 's'); t2 = np.datetime64(t['times'][ks.max()]) + np.timedelta64(1, 's')
    if ks.min() > 0: t1 = min(t1, np.datetime64(t['times'][ks.min() - 1]))
    ekt = EKT.get(u, np.array([], 'datetime64[ns]')); return int(((ekt >= t1) & (ekt <= t2)).sum()), chars
old = pd.read_csv(os.path.join(HERE, '..', 'results', 'r9_audit_sample.csv'))[['u', 'i', 'j']].values.tolist()
for ver in ('var', 'lab'):   # the R9b samples (token rule and n-gram rule), rebuilt with their seed
    rr = pd.DataFrame([(u, c, i, j) for u in KEYU for c, i, j in runs_of(TK[u][ver], 'TX')], columns=['u', 'cls', 'i', 'j'])
    for c in 'TX': old += rr[rr.cls == c].sample(frac=1, random_state=2091).groupby('u').head(3).head(50)[['u', 'i', 'j']].values.tolist()
OLD = {}
for u, i, j in old: OLD.setdefault(int(u), []).append((int(i), int(j)))
def is_old(u, i, j): return any(i < b and a < j for a, b in OLD.get(u, []))
cand = []
for ver in ('var', 'lab'):
    for u in KEYU:
        for c, i, j in runs_of(TK[u][ver], 'TX'):
            if not is_old(u, i, j): cand.append((ver, c, u, i, j))
cand = pd.DataFrame(cand, columns=['stratum_rule', 'cls', 'u', 'i', 'j'])
samp = []
for ver in ('var', 'lab'):
    for c in 'TX':
        x = cand[(cand.stratum_rule == ver) & (cand.cls == c)].sample(frac=1, random_state=2171).groupby('u').head(3).head(50); samp.append(x)
samp = pd.concat(samp).drop_duplicates(['u', 'i', 'j']).reset_index(drop=True)
ver_ = []
for r in samp.itertuples():
    k, ch = keys_verdict(r.u, r.i, r.j); lt = Counter(TK[r.u]['var'][r.i:r.j]).most_common(1)[0][0]; ln = Counter(TK[r.u]['lab'][r.i:r.j]).most_common(1)[0][0]
    ver_.append(dict(keys=k, chars=ch, typed=(k is not None and k >= .5 * ch), not_typed=(k is None or k < .2 * ch), label_token_rule=lt, label_ngram_rule=ln))
samp = pd.concat([samp, pd.DataFrame(ver_)], axis=1)
F23 = {'candidates': int(len(cand)), 'passages': int(len(samp)), 'students': int(samp.u.nunique()),
       'excluded_old_passages': len(old), 'by_stratum': {}, 'pooled': {}}
for ver in ('var', 'lab'):
    for c in 'TX':
        x = samp[(samp.stratum_rule == ver) & (samp.cls == c)]; crit = 'typed' if c == 'T' else 'not_typed'
        F23['by_stratum'][f'{"token" if ver == "var" else "ngram"}_{c}'] = {'passages': int(len(x)), 'students': int(x.u.nunique()),
                                                                              'agree': boot_share(x[crit], x.u, rng)}
for rule, col in (('token', 'label_token_rule'), ('ngram', 'label_ngram_rule')):
    for c in 'TX':
        x = samp[samp[col] == c]; crit = 'typed' if c == 'T' else 'not_typed'
        F23['pooled'][f'{rule}_{c}'] = {'passages': int(len(x)), 'students': int(x.u.nunique()), 'agree': boot_share(x[crit], x.u, rng)}
F23['rules_disagree_passages'] = int((samp.label_token_rule != samp.label_ngram_rule).sum())
out['F23_new_audit_T_X'] = F23
samp.to_csv(os.path.join(HERE, '..', 'results', 'r17_audit_sample.csv'), index=False)
log('F23', {k: v['agree'] for k, v in F23['pooled'].items()})

# ---------- F-24 keystroke support of every echo passage of the 70 students, by route and entry size
erows = []
for u in KEYU:
    t = TK[u]; F = t['F']
    for c, i, j in runs_of(t['lab'], 'E'):
        pg = set(F_ for F_ in t['FG'][i:max(i, j - N + 1)])
        rts = ER[u][i:j]; route = 'prompt' if (rts == 'prompt').any() else 'editor'
        big = (t['te'][i:j] == 2).mean() > .5
        tes = [t['ted'][t['FGs'][g]] for g in pg if t['FGs'][g] in t['ted']]
        if not tes: continue
        t1 = np.datetime64(min(tes)) - np.timedelta64(3, 's'); t2 = np.datetime64(max(tes)) + np.timedelta64(1, 's')
        chars = len(' '.join(F[i:j])); ekt = EKT.get(u, np.array([], 'datetime64[ns]')); keys = int(((ekt >= t1) & (ekt <= t2)).sum())
        eps = EPS.get(u, np.array([], 'datetime64[ns]')); pasted = bool(((eps >= t1 - np.timedelta64(5, 's')) & (eps <= t2)).sum())
        typed = keys >= .5 * chars and not pasted
        if route == 'prompt':
            tp = np.datetime64(min(t['tpr'][g] for g in pg if g in t['tpr'])); w0 = tp - np.timedelta64(600, 's')
            ckt = CKT.get(u, np.array([], 'datetime64[ns]')); cps = CPS.get(u, np.array([], 'datetime64[ns]'))
            ev = bool(((cps >= w0) & (cps <= tp)).sum()) or int(((ckt >= w0) & (ckt <= tp)).sum()) >= .5 * chars
        else:
            tb = min(t['tbot'][g] for g in pg if g in t['tbot']); ev = bool(np.datetime64(min(tes)) < np.datetime64(tb)) and (typed or pasted or keys > 0)
        erows.append(dict(u=u, route=route, large=bool(big), evidence=bool(ev), tokens=j - i))
er = pd.DataFrame(erows); F24 = {'passages': int(len(er)), 'students': int(er.u.nunique()), 'all': boot_share(er.evidence, er.u, rng)}
for (rt, lg), x in er.groupby(['route', 'large']):
    F24[f'{rt}_{"large" if lg else "small"}'] = {'passages': int(len(x)), 'students': int(x.u.nunique()), 'tokens': int(x.tokens.sum()),
                                                'evidence': boot_share(x.evidence, x.u, rng)}
for rt, x in er.groupby('route'): F24[rt] = {'passages': int(len(x)), 'students': int(x.u.nunique()), 'evidence': boot_share(x.evidence, x.u, rng)}
for lg, x in er.groupby('large'): F24['large' if lg else 'small'] = {'passages': int(len(x)), 'students': int(x.u.nunique()), 'evidence': boot_share(x.evidence, x.u, rng)}
out['F24_echo_keystroke_census'] = F24
log('F24', {k: v for k, v in F24.items() if isinstance(v, dict)})

# ---------- F-25 where the large-entry echo came from
# (a) whether the echo text entered the editor after a reply that contained it (so that it could have been taken from the
# reply, although the writer's side had it first); (b) for students with logged pastes, whether the paste that carried it
# also carried chatbot-originated text, i.e. 8-grams of the student's own earlier replies that no earlier turn contained;
# (c) whether it occurs in a reply given to another student.
pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms')
PGT = {u: [(np.datetime64(t_), set(grams(tok(s), N))) for t_, s in zip(x.t, x.txt)] for u, x in pa.groupby('user_id')}
OWNR = {}
for u, x in d[d.user_id.isin(users)].groupby('user_id'):
    for s in x.chatbot_utterance_text.astype(str):
        for g in set(grams(tok(s), N)): OWNR.setdefault(g, set()).add(u)
cnt = Counter(); cnt_p = Counter()
for u in users:
    t = TK[u]; F = t['F']; m = (t['lab'] == 'E') & (t['te'] == 2)
    if not m.any(): continue
    after = np.zeros(len(F), bool); coh = np.zeros(len(F), bool)
    for i, g in enumerate(t['FG']):
        if t['cls'][g] == 'E' and g in t['tbot']:
            te_ = t['ted'].get(t['FGs'][g])
            if te_ is not None and pd.Timestamp(te_) >= pd.Timestamp(t['tbot'][g]): after[i:i + N] = True
        if OWNR.get(g, set()) - {u}: coh[i:i + N] = True
    cnt['E_large'] += int(m.sum()); cnt['entered_after_a_reply_containing_it'] += int((m & after).sum()); cnt['in_cohort_reply'] += int((m & coh).sum())
    if u in PGT:
        botfirst = {g for g, tb in t['tbot'].items() if not (g in t['tpr'] and t['tpr'][g] < tb)}
        inp = np.zeros(len(F), bool); inp_bot = np.zeros(len(F), bool)
        for tp_, pg in PGT[u]:
            carries = any(g in botfirst and np.datetime64(t['tbot'][g]) <= tp_ for g in pg)
            for i, g in enumerate(t['FG']):
                if g in pg:
                    inp[i:i + N] = True
                    if carries: inp_bot[i:i + N] = True
        cnt_p['E_large'] += int(m.sum()); cnt_p['in_logged_paste'] += int((m & inp).sum())
        cnt_p['paste_also_carries_chatbot_first_text'] += int((m & inp_bot).sum()); cnt_p['entered_after_a_reply_containing_it'] += int((m & after).sum())
out['F25_large_echo_sources'] = {'all_students': {k: (v / cnt['E_large'] if k != 'E_large' else v) for k, v in cnt.items()},
                                 'students_with_logged_pastes': {k: (v / cnt_p['E_large'] if k != 'E_large' else v) for k, v in cnt_p.items()},
                                 'n_students_with_pastes': len(PGT)}
log('F25', out['F25_large_echo_sources'])

# ---------- F-26 peer threshold as a count and as a share of the cohort
NU = len(users); F26 = {}
for name, k in [('count_3', 3), ('count_5', 5), ('count_10', 10), ('share_1pct', int(np.ceil(.01 * (NU - 1)))),
                ('share_2pct', int(np.ceil(.02 * (NU - 1)))), ('share_5pct', int(np.ceil(.05 * (NU - 1))))]:
    shared = lambda g: g in H[8] or Cc[8][g] - 1 >= k
    s_share, c_lost = [], 0
    for u in users:
        t = TK[u]; F = t['F']; m = np.zeros(len(F), bool)
        for i, g in enumerate(t['FG']):
            if t['cls'][g] in ('C', 'S') and shared(g) and not (g in H[8]): m[i:i + N] = True
        s_share.append(float(m.mean()) if len(F) else 0.0)
    F26[name] = {'peer_threshold_students': k, 'mean_share_of_proposal_peer_recurrent_C_or_S': float(np.mean(s_share))}
out['F26_peer_threshold'] = F26; out['F26_peer_threshold']['cohort_size'] = NU

# ---------- F-27 chance coverage of a proposal by the replies of other students, n = 5 to 12
rngc = np.random.default_rng(2172); F27 = {}
RT = {u: [tok(s) for s in x.chatbot_utterance_text.astype(str)] for u, x in d[d.user_id.isin(users)].groupby('user_id')}
others = {u: rngc.choice([v for v in users if v != u], 20, replace=False) for u in users}
for n in range(5, 13):
    cov_null, cov_own = [], []
    for u in users:
        F = P[u]; G_ = grams(F, n)
        if not G_: continue
        def cover(S_):
            m = np.zeros(len(F), bool)
            for i, g in enumerate(G_):
                if g in S_: m[i:i + n] = True
            return m.mean()
        own = {g for r in RT.get(u, []) for g in grams(r, n)}; cov_own.append(cover(own))
        cov_null.append(np.mean([cover({g for r in RT.get(v, []) for g in grams(r, n)}) for v in others[u]]))
    cov_null = np.array(cov_null)
    F27[n] = {'null_mean': float(cov_null.mean()), 'null_median': float(np.median(cov_null)), 'null_share_students_above_1pct': float((cov_null > .01).mean()),
              'own_mean': float(np.mean(cov_own))}
    log('F27', n, F27[n])
out['F27_null_coverage_by_n'] = F27

out['seconds'] = time.perf_counter() - t00
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r17_flora.json'), 'w'), indent=1, default=float)
log('done')
