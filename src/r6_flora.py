"""Plan R6, FLoRA (PREREG_R6.md G1-G5): degraded logs (dialogue only, thinned snapshots), parameter choices
(chance baseline by n, entry threshold tau, effect of every variant on the main conclusions), subdivision of X by entry
evidence, origin of logged pastes, precision of the grade estimates.
Usage: python -I src/r6_flora.py DATA_DIR -> results/r6_flora.json"""
import sys, re, json, warnings, time
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2060); NB = 2000
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
an = pd.read_pickle(f'{D}/annot.pkl'); tr = pd.read_pickle(f'{D}/trace.pkl')
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); del tr
pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms'); pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
users = sorted(set(d.user_id) & set(p.index))
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
users_all = users; users = users[:LIMIT] if LIMIT else users
score = p.loc[users, 'proposal_score'].values.astype(float)
NS = (5, 8, 12)
H = {n: set(g for t in an.highlight_text.dropna() for g in grams(tok(t), n)) for n in NS}
Cc = {n: Counter(g for u in users_all for g in set(grams(P[u], n))) for n in NS}
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
WG = {u: g for u, g in w.groupby('user_id')}
U1 = np.load('results/U_R1.npy')[:len(users)]
THIN = (5, 15, 60, 300); TAUS = (1, 2, 5, 10, 20, 50)

def dialogue(u, n):
    x = d[d.user_id == u]; tbot, tpr, plen = {}, {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s), n): tbot.setdefault(g, pd.Timestamp(t))
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(s), n): tpr.setdefault(g, pd.Timestamp(t))
    return tbot, tpr

def scan(enc, times, idx, n, need, want_seen=False):
    """Entry of each n-gram of A (int-encoded tuples in need): time, insertion count of the whole transition,
    size of the inserted segment containing it, transition index. Optionally first-seen time of every n-gram."""
    need = set(need); ted, ins, seg, tid, seen = {}, {}, {}, {}, {}
    prev = np.zeros(0, np.int32)
    for k, i in enumerate(idx):
        if not need and not want_seen: break
        cur = enc[i]; L = len(cur)
        if len(prev) == 0:
            segs = [(0, L)] if L else []; lo, hi = 0, L - n + 1
        else:
            m = min(len(prev), L); ne = np.flatnonzero(prev[:m] != cur[:m]); pre = int(ne[0]) if len(ne) else m
            mm = min(len(prev), L) - pre; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
            a, b = prev[pre:len(prev) - suf], cur[pre:L - suf]
            if len(a) == 0 and len(b) == 0: prev = cur; continue
            if len(b) == 0: segs = []
            elif len(a) == 0 or len(b) == 1: segs = [(pre, pre + len(b))]
            else:
                segs = [(pre + j1, pre + j2) for op, i1, i2, j1, j2 in SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes() if op in ('insert', 'replace')]
            lo, hi = max(0, pre - n + 1), min(L - n + 1, max(L - suf, pre + 1))
        tot = sum(j2 - j1 for j1, j2 in segs); t = times[i]
        for s in range(max(lo, 0), max(hi, 0)):
            g = tuple(cur[s:s + n].tolist())
            if want_seen and g not in seen: seen[g] = t
            if g in need:
                best = (0, 0)
                for j1, j2 in segs:
                    ov = min(s + n, j2) - max(s, j1)
                    if ov > best[0]: best = (ov, j2 - j1)
                ted[g] = t; ins[g] = tot; seg[g] = best[1]; tid[g] = k; need.discard(g)
        prev = cur
    return dict(ted=ted, ins=ins, seg=seg, tid=tid, seen=seen)

def classify(FGs, tbot, tpr, sc, n, tau=5, delta=pd.Timedelta(0), mode='ins', k_peer=5):
    cls = {}
    for g, gi in FGs.items():
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]; shared = g in H[n] or Cc[n][g] - 1 >= k_peer
        te = sc['ted'].get(gi)
        if bot and not echo and not shared and te is not None and tbot[g] <= te + delta: cls[g] = 'C'
        elif bot and (echo or (te is not None and te < tbot[g])): cls[g] = 'E'
        elif shared: cls[g] = 'S'
        elif te is None: cls[g] = 'X'
        else: cls[g] = 'T' if sc[mode][gi] <= tau else 'X'
    return cls

def project(F, n, cls, order='CESTX'):
    rank = {c: i for i, c in enumerate(order)}; best = np.full(len(F), 99)
    for i, g in enumerate(grams(F, n)): best[i:i + n] = np.minimum(best[i:i + n], rank[cls[g]])
    return np.array([order[b] if b < 99 else 'X' for b in best])

def majority(F, n, cls, order='CESTX'):
    votes = [Counter() for _ in F]; rank = {c: i for i, c in enumerate(order)}
    for i, g in enumerate(grams(F, n)):
        for j in range(i, i + n): votes[j][cls[g]] += 1
    return np.array([min(v, key=lambda c: (-v[c], rank[c])) if v else 'X' for v in votes])

LAB = defaultdict(dict); AUX = {}
t0 = time.time()
for k, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in (g_.writing_content if g_ is not None else [])]
    F = P[u]; Fi = [vocab.setdefault(x, len(vocab)) for x in F]
    full = list(range(len(enc)))
    for n in NS:
        tbot, tpr = dialogue(u, n); FG = grams(F, n); FGs = {g: tuple(Fi[i:i + n]) for i, g in enumerate(FG)}
        sc = scan(enc, times, full, n, set(FGs.values()), want_seen=(n == 8 and u in set(pa.user_id)))
        if n == 8:
            cls = classify(FGs, tbot, tpr, sc, 8); LAB[u]['R4'] = project(F, 8, cls)
            for tau in TAUS:
                if tau != 5: LAB[u][f'tau{tau}'] = project(F, 8, classify(FGs, tbot, tpr, sc, 8, tau=tau))
            LAB[u]['delta1h'] = project(F, 8, classify(FGs, tbot, tpr, sc, 8, delta=pd.Timedelta(hours=1)))
            for kp in (3, 10): LAB[u][f'peer{kp}'] = project(F, 8, classify(FGs, tbot, tpr, sc, 8, k_peer=kp))
            LAB[u]['EC'] = project(F, 8, cls, 'ECSTX'); LAB[u]['MAJ'] = majority(F, 8, cls)
            LAB[u]['SEG'] = project(F, 8, classify(FGs, tbot, tpr, sc, 8, mode='seg'))
            # dialogue only
            dcls = {}
            for g in FG:
                bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]; shared = g in H[8] or Cc[8][g] - 1 >= 5
                dcls[g] = 'E' if echo else 'S' if shared else 'C' if bot else 'N'
            LAB[u]['DIAL'] = project(F, 8, dcls, 'CESN')
            AUX[u] = dict(cls=cls, FGs=FGs, sc=sc, tbot=tbot, tpr=tpr, vocab=vocab)
            for dt in THIN:
                keep, last = [], None
                for i, t in enumerate(times):
                    if last is None or (t - last).total_seconds() >= dt: keep.append(i); last = t
                if full and keep[-1] != full[-1]: keep.append(full[-1])
                st = scan(enc, times, keep, 8, set(FGs.values()))
                for tau in (5, 20): LAB[u][f'thin{dt}_tau{tau}'] = project(F, 8, classify(FGs, tbot, tpr, st, 8, tau=tau))
                AUX[u][f'nsnap{dt}'] = len(keep)
            AUX[u]['nsnap'] = len(full)
        else:
            LAB[u][f'n{n}'] = project(F, n, classify(FGs, tbot, tpr, sc, n))
    if k % 25 == 0: print(k, round(time.time() - t0), flush=True)
del WG, w

out = {'n_students': len(users)}
Ush = lambda key: np.array([(LAB[u][key] == 'C').mean() for u in users])
out['check'] = {'max_abs_diff_C_vs_Ustar': float(np.abs(Ush('R4') - U1).max()), 'mean_abs_diff': float(np.abs(Ush('R4') - U1).mean()),
                'R4_class_means': {c: float(np.mean([(LAB[u]['R4'] == c).mean() for u in users])) for c in 'CESTX'}}
tokens_total = sum(len(P[u]) for u in users)
def hc3(v):
    z = (v - v.mean()) / v.std(); f = sm.OLS(score, sm.add_constant(z)).fit(cov_type='HC3')
    return [float(f.params[1]), [float(c) for c in f.conf_int()[1]]]
def summ(key, ref='R4'):
    U = Ush(key); o = {'U_median': float(np.median(U)), 'ge50': float((U >= .5).mean()), 'ge25': float((U >= .25).mean()),
                       'rho_with_Ustar': float(stats.spearmanr(U, U1)[0]), 'grade_slope_per_sd': hc3(U)}
    for c in 'CESTXN': o[f'{c}_mean'] = float(np.mean([(LAB[u][key] == c).mean() for u in users]))
    o['tokens_changed_vs_R4'] = sum(int((LAB[u][key] != LAB[u][ref]).sum()) for u in users) / tokens_total
    return o
# G2(c) every variant
out['variants'] = {key: summ(key) for key in ['R4', 'n5', 'n12', 'delta1h', 'peer3', 'peer10', 'EC', 'MAJ', 'SEG'] + [f'tau{t}' for t in TAUS if t != 5]}
# G1(a) dialogue only
o = summ('DIAL'); R4C = sum(int((LAB[u]['R4'] == 'C').sum()) for u in users); R4E = sum(int((LAB[u]['R4'] == 'E').sum()) for u in users)
o['R4_C_tokens_labelled_C'] = sum(int(((LAB[u]['R4'] == 'C') & (LAB[u]['DIAL'] == 'C')).sum()) for u in users) / R4C
o['R4_E_tokens_labelled_E'] = sum(int(((LAB[u]['R4'] == 'E') & (LAB[u]['DIAL'] == 'E')).sum()) for u in users) / R4E
o['R4_E_tokens_labelled_C'] = sum(int(((LAB[u]['R4'] == 'E') & (LAB[u]['DIAL'] == 'C')).sum()) for u in users) / R4E
o['DIAL_C_tokens_that_are_R4_C'] = sum(int(((LAB[u]['R4'] == 'C') & (LAB[u]['DIAL'] == 'C')).sum()) for u in users) / max(1, sum(int((LAB[u]['DIAL'] == 'C').sum()) for u in users))
out['G1a_dialogue_only'] = o
# G1(b) thinning
def kappa(a, b):
    m = np.isin(a, ['T', 'X']) & np.isin(b, ['T', 'X'])
    if m.sum() == 0: return None
    a, b = a[m] == 'X', b[m] == 'X'; po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else None
thin = {'snapshots_total': int(sum(AUX[u]['nsnap'] for u in users))}
for dt in THIN:
    for tau in (5, 20):
        key = f'thin{dt}_tau{tau}'; o = summ(key)
        o['snapshots_kept'] = int(sum(AUX[u][f'nsnap{dt}'] for u in users))
        o['kappa_TX_vs_R4'] = kappa(np.concatenate([LAB[u]['R4'] for u in users]), np.concatenate([LAB[u][key] for u in users]))
        thin[key] = o
out['G1b_thinned'] = thin
# G2(a) chance baseline by n
cb = {}
for n in NS:
    key = 'R4' if n == 8 else f'n{n}'; Un = Ush(key)
    botf = {}
    for u in users:
        x = d[d.user_id == u]; botf[u] = {g for t in x.chatbot_utterance_text for g in grams(tok(t), n)} - H[n]
    above, nulmed = [], []
    for i, u in enumerate(users):
        FG = grams(P[u], n); others = [v for v in users if v != u]; nul = []
        for v in rng.choice(others, 200):
            S_ = botf[v]; m = np.zeros(len(P[u]), bool)
            for j, g in enumerate(FG):
                if g in S_ and Cc[n][g] - 1 < 5: m[j:j + n] = True
            nul.append(m.mean())
        nul = np.array(nul); nulmed.append(float(np.median(nul))); above.append(Un[i] > np.percentile(nul, 99))
    above = np.array(above, float)
    cb[f'n{n}'] = {'null_median_coverage_median': float(np.median(nulmed)), 'null_median_coverage_mean': float(np.mean(nulmed)),
                   'share_above_own_null99': float(above.mean()), 'ci': [float(np.percentile([rng.choice(above, len(above)).mean() for _ in range(NB)], q)) for q in (2.5, 97.5)],
                   'U_median': float(np.median(Un))}
out['G2a_chance_by_n'] = cb
# G3 subdivision of X (gram level, R4 labels)
pu = set(pa.user_id); win = pd.Timedelta(seconds=10)
cnt = Counter(); cntp = Counter()
for u in users:
    A = AUX[u]; sc = A['sc']; cls = A['cls']; Ctid = {sc['tid'][A['FGs'][g]] for g, c in cls.items() if c == 'C' and A['FGs'][g] in sc['tid']}
    pt = np.array(pa[pa.user_id == u].t.values, dtype='datetime64[ns]') if u in pu else None
    for g, c in cls.items():
        if c != 'X': continue
        gi = A['FGs'][g]; cnt['X'] += 1
        if gi not in sc['ted']: cnt['never_in_snapshots'] += 1; continue
        co = sc['tid'][gi] in Ctid; big = sc['seg'][gi] > 50
        cnt['coentered_with_C'] += co; cnt['segment_gt50'] += big; cnt['segment_6_50'] += (5 < sc['seg'][gi] <= 50); cnt['segment_le5'] += sc['seg'][gi] <= 5
        if pt is not None:
            cntp['X'] += 1; te = np.datetime64(sc['ted'][gi], 'ns')
            lp = bool(len(pt)) and bool((np.abs(pt - te) <= np.timedelta64(10, 's')).any())
            cntp['logged_paste'] += lp; cntp['logged_paste_and_coentered_C'] += lp and co; cntp['no_paste_not_coentered'] += (not lp) and (not co)
out['G3_X_subdivision'] = {'all_students': {k: (v / cnt['X'] if k != 'X' else v) for k, v in cnt.items()},
                           'students_with_logged_pastes': {k: (v / cntp['X'] if k != 'X' else v) for k, v in cntp.items()} | {'n_students': len(pu & set(users))}}
# G4 origin of logged pastes
cat = Counter(); share_rows = []; Cconf = Counter(); Crs = []; rshares = defaultdict(list)
for u in sorted(pu & set(users)):
    A = AUX[u]; seen = A['sc']['seen']; tbot, tpr = A['tbot'], A['tpr']; vocab_of = {}
    # seen holds int-encoded grams; rebuild the student's vocabulary mapping from the final-text encoding
    rev = A['vocab']
    pastes = pa[pa.user_id == u].sort_values('t'); pcls = []
    for t, txt in zip(pastes.t, pastes.txt):
        G = set(grams(tok(txt), 8))
        if not G: pcls.append((t, None, G)); continue
        a_ = np.mean([g in tbot and tbot[g] < t for g in G]); b_ = np.mean([g in tpr and tpr[g] < t for g in G])
        enc = [tuple(rev.get(x, -1) for x in g) for g in G]
        c_ = np.mean([(e in seen and seen[e] < t - pd.Timedelta(seconds=2)) for e in enc])
        none = np.mean([not ((g in tbot and tbot[g] < t) or (g in tpr and tpr[g] < t) or (e in seen and seen[e] < t - pd.Timedelta(seconds=2))) for g, e in zip(G, enc)])
        k_ = 'reply' if a_ >= .8 else 'internal' if c_ >= .8 else 'external' if none >= .8 else 'mixed'
        cat[k_] += 1; share_rows.append((a_, b_, c_, none)); pcls.append((t, k_, G)); rshares[u].append(a_)
    # C tokens confirmed by a logged paste: class of the earliest paste containing a covering C gram
    F = P[u]; lab = LAB[u]['R4']; FG = grams(F, 8)
    for j in np.flatnonzero(lab == 'C'):
        kk = None
        for t, k_, G in pcls:
            if k_ is None: continue
            if any(FG[s] in G for s in range(max(0, j - 7), min(len(FG), j + 1))): kk = k_; break
        Cconf[kk or 'not_in_logged_paste'] += 1
        if kk is not None: Crs.append(next(a for (t_, k_, G_), a in zip([x for x in pcls if x[1] is not None], rshares[u]) if k_ == kk and any(FG[s] in G_ for s in range(max(0, j - 7), min(len(FG), j + 1)))))
sr = np.array(share_rows)
out['G4_paste_origin'] = {'pastes_ge8_tokens': int(sum(cat.values())), 'class_counts': dict(cat),
                          'mean_shares': {'earlier_reply': float(sr[:, 0].mean()), 'earlier_prompt': float(sr[:, 1].mean()), 'earlier_editor': float(sr[:, 2].mean()), 'none': float(sr[:, 3].mean())},
                          'C_tokens_by_confirming_paste_class': {k: v / sum(Cconf.values()) for k, v in Cconf.items()}, 'C_tokens_total': int(sum(Cconf.values())),
                          'C_confirmed_tokens_paste_reply_share': {'median': float(np.median(Crs)), 'share_ge80': float(np.mean(np.array(Crs) >= .8)), 'share_lt50': float(np.mean(np.array(Crs) < .5))}}
# G5 precision of grade estimates
sdU = float(U1.std()); g5 = {'sd_Ustar': sdU, 'grade_sd': float(score.std())}
z = (U1 - U1.mean()) / U1.std(); f = sm.OLS(score, sm.add_constant(z)).fit(cov_type='HC3')
g5['unadjusted'] = {'per_sd': float(f.params[1]), 'se': float(f.bse[1]), 'mde_per_sd': float(2.80 * f.bse[1]), 'per_10pp': float(f.params[1] * .1 / sdU),
                    'ci_per_10pp': [float(c * .1 / sdU) for c in f.conf_int()[1]], 'mde_per_10pp': float(2.80 * f.bse[1] * .1 / sdU),
                    'mde_in_grade_sd': float(2.80 * f.bse[1] / score.std())}
out['G5_precision'] = g5
json.dump(out, open('results/r6_flora.json' if not LIMIT else '/tmp/r6_test.json', 'w'), indent=1); print(json.dumps(out, indent=1))
