"""Analysis log R18, F-31 to F-34 (FLoRA).
F-31 when FLoRA logged a reply: latency between the logged ask time and the logged reply time against reply length,
and reply 8-grams that first reached the editor between the ask and the logged reply time (text the writer could only
have had from a reply shown before it was logged), against the same span after the logged reply time;
F-32 the attribution with the clock tolerance delta = +5, +30, +60 and +120 s;
F-33 the distribution of snapshot transitions by the number of inserted tokens, and for the 70 keystroke students the
share of transitions in each size band that were typed (printable editor keystrokes from the previous snapshot, at most
3 s before, to 1 s after the snapshot, at least half of the inserted characters; the window of the R9 audit);
F-34 the shared-source rule without the peer criterion for 8-grams that the student's own chatbot produced first.
The labels at delta = 0 are those of the attribution (src/r9b_flora_token_entry.py definitions); nothing changes in them.
Usage: python -I src/r18_flora.py DATA_DIR -> results/r18_flora.json"""
import sys, os, json, time
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1]
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)
src = open(os.path.join(HERE, 'r9b_flora_token_entry.py')).read()
cut = src.index('# ======================= R9b body')
sys.argv = [os.path.join(HERE, 'r9b_flora_token_entry.py'), D]
G = {'__name__': 'r9b_defs', '__file__': os.path.join(HERE, 'r9b_flora_token_entry.py')}
exec(compile(src[:cut], 'r9b_flora_token_entry.py', 'exec'), G)
d, users, P, WG, H, Cc, tok, grams, scan, classify, project = (G[k] for k in
    ('d', 'users_all', 'P', 'WG', 'H', 'Cc', 'tok', 'grams', 'scan', 'classify', 'project'))
N = 8; NB = 2000; rng = np.random.default_rng(2181)
out = {}

# ---------- F-31 (a) latency against reply length
dd = d.copy()
dd['lat'] = (pd.to_datetime(dd.chatbot_utterance_time) - pd.to_datetime(dd.user_ask_time)).dt.total_seconds()
dd['ntok'] = dd.chatbot_utterance_text.map(lambda s: len(tok(s)))
x = dd[(dd.lat > 0) & (dd.lat < 600)]
qr = sm.QuantReg(x.lat.values, sm.add_constant(x.ntok.values.astype(float))).fit(q=.5)
bands = [(0, 50), (50, 150), (150, 300), (300, 500), (500, 10 ** 6)]
out['F31a_latency'] = {
    'replies': int(len(dd)), 'replies_latency_0_600s': int(len(x)),
    'latency_s_quantiles': {q: float(dd.lat.quantile(q)) for q in (.05, .25, .5, .75, .95)},
    'spearman_latency_length': float(stats.spearmanr(x.ntok, x.lat)[0]),
    'median_regression_intercept_s': float(qr.params[0]), 'median_regression_s_per_token': float(qr.params[1]),
    'median_regression_ci': qr.conf_int().tolist(),
    'by_length_band': {f'{a}-{b}': {'replies': int(((x.ntok >= a) & (x.ntok < b)).sum()),
                                    'median_latency_s': float(x.lat[(x.ntok >= a) & (x.ntok < b)].median())} for a, b in bands}}
log('F31a', out['F31a_latency']['median_regression_s_per_token'])

# ---------- per student: first editor time of every 8-gram, labels under delta variants, transitions
DELTAS = [0, 5, 30, 60, 120]
pre_rows = []; gap_rows = []; U = {k: [] for k in DELTAS}; Csh = {k: [] for k in DELTAS}; Esh = {k: [] for k in DELTAS}
U_s = []; trans = []
tr = pd.read_pickle(f'{D}/trace.pkl'); tr['t'] = pd.to_datetime(tr.log_trace_time, unit='ms')
ek = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy(); del tr
key = ek.event_value.astype(str).str.replace('KEY:::', '', regex=False).str.split('---').str[0]
EKT = {u: np.sort(g_.t.values[key[g_.index].str.len().eq(1).values]) for u, g_ in ek.groupby('user_id')}
KEYU = set(EKT) & set(users)

def dialogue_full(u):
    x_ = d[d.user_id == u]
    return list(zip(pd.to_datetime(x_.user_ask_time), pd.to_datetime(x_.chatbot_utterance_time),
                    x_.user_utterance_text.astype(str), x_.chatbot_utterance_text.astype(str)))

for kk, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; vocab = {}
    contents = list(g_.writing_content) if g_ is not None else []
    enc = [np.array([vocab.setdefault(t_, len(vocab)) for t_ in tok(c)], np.int32) for c in contents]
    F = P[u]; Fi = np.array([vocab.setdefault(t_, len(vocab)) for t_ in F], np.int32)
    tbot, tpr = G['dialogue'](u, N); FG = grams(F, N); FGs = {g: tuple(Fi[i:i + N].tolist()) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), N, set(FGs.values()), want_seen=True)
    seen = sc['seen']
    # F-32 delta variants
    for dl in DELTAS:
        cls = classify(FGs, tbot, tpr, sc, N, delta=pd.Timedelta(seconds=dl)); lab = project(F, N, cls)
        U[dl].append(float((lab == 'C').mean()) if len(F) else 0.0); Esh[dl].append(float((lab == 'E').mean()) if len(F) else 0.0)
    # F-34 peer criterion not applied to 8-grams the student's own chatbot produced first
    Hs = H[N]; Cn = Cc[N]
    cls_s = {}
    for g, gi in FGs.items():
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]
        shared = g in Hs or (Cn[g] - 1 >= 5 and not (bot and not echo))
        te = sc['ted'].get(gi)
        if bot and not echo and not shared and te is not None and tbot[g] <= te: cls_s[g] = 'C'
        elif bot and (echo or (te is not None and te < tbot[g])): cls_s[g] = 'E'
        elif shared: cls_s[g] = 'S'
        else: cls_s[g] = 'X'
    U_s.append(float((project(F, N, cls_s) == 'C').mean()) if len(F) else 0.0)
    # F-31 (b) reply 8-grams first reaching the editor between the ask and the logged reply time
    dia = dialogue_full(u); first_turn = {}; first_reply = {}
    for a_, r_, q_, s_ in dia:
        for g in grams(tok(q_), N): first_turn.setdefault(g, a_)
    for j, (a_, r_, q_, s_) in enumerate(dia):
        lat = (r_ - a_).total_seconds()
        new = set()
        for g in set(grams(tok(s_), N)):
            if g in first_reply and first_reply[g] < r_: continue
            if g in first_turn and first_turn[g] <= a_: continue
            if any(t_ not in vocab for t_ in g): continue
            gi = tuple(vocab[t_] for t_ in g); ts = seen.get(gi)
            if ts is None: continue
            ts = pd.Timestamp(ts)
            if ts < a_: continue
            new.add(g); gap_rows.append((u, j, (ts - r_).total_seconds(), lat))
        for g in grams(tok(s_), N): first_reply.setdefault(g, r_)
        if lat > 0:
            pre = sum(1 for (uu, jj, gp, _) in gap_rows[-len(new):] if gp < 0) if new else 0
            post = sum(1 for (uu, jj, gp, _) in gap_rows[-len(new):] if 0 <= gp < lat) if new else 0
            pre_rows.append((u, j, lat, len(tok(s_)), pre, post))
    # F-33 transitions: inserted tokens and keystrokes since the previous snapshot
    if u in KEYU and len(enc) > 1:
        kt = EKT[u]; tv = np.array(times, dtype='datetime64[ns]')
        for k in range(1, len(enc)):
            prev, cur = enc[k - 1], enc[k]; L = len(cur)
            m = min(len(prev), L); ne = np.flatnonzero(prev[:m] != cur[:m]); pre_ = int(ne[0]) if len(ne) else m
            mm = min(len(prev), L) - pre_; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
            ins = L - suf - pre_; dele = len(prev) - suf - pre_
            if ins <= 0: continue
            itoks = [t_ for t_ in tok(contents[k])][pre_:L - suf]
            nch = sum(len(t_) for t_ in itoks) + max(len(itoks) - 1, 0)
            nk = int(np.searchsorted(kt, tv[k] + np.timedelta64(1, 's'), 'right') - np.searchsorted(kt, max(tv[k - 1], tv[k] - np.timedelta64(3, 's')), 'right'))
            trans.append((u, ins, nch, nk))
    if kk % 50 == 0: log(kk)

# ---------- F-31 (b) summary
pr = pd.DataFrame(pre_rows, columns=['u', 'j', 'lat', 'ntok', 'pre', 'post'])
gp = pd.DataFrame(gap_rows, columns=['u', 'j', 'gap', 'lat'])
def boot_mean_by_student(df, col):
    us = df.u.unique(); grp = {k: v[col].values for k, v in df.groupby('u')}; bs = []
    for _ in range(NB):
        s = rng.choice(us, len(us)); bs.append(np.concatenate([grp[k] for k in s]).mean())
    return [float(df[col].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
pr['any_pre'] = (pr.pre > 0).astype(float); pr['any_post'] = (pr.post > 0).astype(float)
bins = [-1e9, -60, -30, -10, -5, 0, 5, 30, 60, 300, 3600, 1e12]
lab_b = ['<-60', '[-60,-30)', '[-30,-10)', '[-10,-5)', '[-5,0)', '[0,5)', '[5,30)', '[30,60)', '[60,300)', '[300,3600)', '>=3600']
h = pd.cut(gp.gap, bins, right=False, labels=lab_b).value_counts().reindex(lab_b)
out['F31b_reply_ngrams_before_logged_time'] = {
    'replies_scored': int(len(pr)), 'students': int(pr.u.nunique()),
    'reply_8grams_entering_editor_after_ask': int(len(gp)),
    'share_of_those_before_logged_reply_time': float((gp.gap < 0).mean()) if len(gp) else None,
    'share_within_logged_latency_after_reply_time': float(((gp.gap >= 0) & (gp.gap < gp.lat)).mean()) if len(gp) else None,
    'replies_with_any_8gram_before_logged_time': boot_mean_by_student(pr, 'any_pre'),
    'replies_with_any_8gram_in_same_span_after': boot_mean_by_student(pr, 'any_post'),
    'students_with_any_8gram_before_logged_time': int(pr[pr.pre > 0].u.nunique()),
    'gap_histogram_s': {k: int(v) for k, v in h.items()}}
log('F31b', out['F31b_reply_ngrams_before_logged_time'])

# ---------- F-32 delta variants
U0 = np.array(U[0])
def boot_share(v, thr=.5):
    v = np.asarray(v); bs = [np.mean(v[rng.integers(0, len(v), len(v))] >= thr) for _ in range(NB)]
    return [float(np.mean(v >= thr)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out['F32_delta'] = {str(dl): {'median_U': float(np.median(U[dl])), 'mean_U': float(np.mean(U[dl])), 'mean_E': float(np.mean(Esh[dl])),
                              'share_at_or_above_half': boot_share(U[dl]), 'rho_with_U': float(stats.spearmanr(U[dl], U0)[0]),
                              'students_changed': int(np.sum(np.abs(np.array(U[dl]) - U0) > 1e-12))} for dl in DELTAS}
log('F32', {k: (round(v['median_U'], 4), v['share_at_or_above_half'][0]) for k, v in out['F32_delta'].items()})

# ---------- F-33 transitions by inserted tokens
T_ = pd.DataFrame(trans, columns=['u', 'ins', 'nch', 'nk'])
T_['typed'] = T_.nk >= 0.5 * T_.nch
bands = [(1, 1), (2, 5), (6, 20), (21, 10 ** 9)]
allins = []
for u, g_ in WG.items():
    if u not in set(users): continue
    prev = []
    for c in g_.writing_content:
        cur = tok(c); L = len(cur); m = min(len(prev), L); pre_ = 0
        while pre_ < m and prev[pre_] == cur[pre_]: pre_ += 1
        suf = 0
        while suf < m - pre_ and prev[len(prev) - 1 - suf] == cur[L - 1 - suf]: suf += 1
        ins = L - suf - pre_
        if ins > 0 or len(prev) - suf - pre_ > 0: allins.append(ins)
        prev = cur
allins = np.array(allins)
out['F33_transitions'] = {
    'transitions_all_students': int(len(allins)),
    'share_by_inserted_tokens_all_students': {'0 (deletion only)': float(np.mean(allins == 0)), **{f'{a}-{b}' if b < 10 ** 9 else f'>{a - 1}': float(np.mean((allins >= a) & (allins <= b))) for a, b in bands}},
    'keystroke_students': int(T_.u.nunique()), 'insertions_keystroke_students': int(len(T_)),
    'typed_share_by_band': {f'{a}-{b}' if b < 10 ** 9 else f'>{a - 1}': {'insertions': int(((T_.ins >= a) & (T_.ins <= b)).sum()),
                                                                      'typed': float(T_.typed[(T_.ins >= a) & (T_.ins <= b)].mean())} for a, b in bands}}
log('F33', out['F33_transitions'])

# ---------- F-34 shared-source variant
Us = np.array(U_s)
out['F34_peer_not_for_own_chatbot_text'] = {'median_U': float(np.median(Us)), 'mean_U': float(np.mean(Us)),
                                             'share_at_or_above_half': boot_share(Us), 'rho_with_U': float(stats.spearmanr(Us, U0)[0]),
                                             'mean_U_difference_pp': float(100 * np.mean(Us - U0))}
log('F34', out['F34_peer_not_for_own_chatbot_text'])
out['check_delta0'] = {'median_U': float(np.median(U0)), 'mean_U': float(np.mean(U0)), 'share_at_or_above_half': float(np.mean(U0 >= .5))}
out['seconds'] = round(time.perf_counter() - t00)
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r18_flora.json'), 'w'), indent=1)
log('done')
