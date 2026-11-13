"""Analysis log R11, P-1 (FLoRA part) and F-1 to F-5: token entry rule as primary, unresolved text, shared-source
n-grams in replies, clock direction, clustering of the keystroke audit. Re-exports the token-level and student-level
class files with the token entry rule (n-gram entry kept as a column).
Usage: python -I src/r11_flora.py DATA_DIR -> results/r11_flora.json, results/flora_token_classes.csv.gz, results/flora_student_classes.csv"""
import sys, re, json, warnings, time
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from scipy import stats, optimize
import statsmodels.api as sm
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2080); NB = 2000
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
an = pd.read_pickle(f'{D}/annot.pkl'); tr = pd.read_pickle(f'{D}/trace.pkl')
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); del tr
pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms'); pa['txt'] = pa.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
users = sorted(set(d.user_id) & set(p.index))
LIMIT = None
P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
users_all = users; users = users[:LIMIT] if LIMIT else users
score = p.loc[users, 'proposal_score'].values.astype(float)
NS = (5, 8, 12)
H = {n: set(g for t in an.highlight_text.dropna() for g in grams(tok(t), n)) for n in NS}
Cc = {n: Counter(g for u in users_all for g in set(grams(P[u], n))) for n in NS}
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
WG = {u: g for u, g in w.groupby('user_id')}
U1 = np.load('results/U_R1.npy')[:len(users)]
NS = (8,)

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



# ======================= R9b body: token-level entry mode =======================

users = users_all
tr = pd.read_pickle(f'{D}/trace.pkl'); tr['t'] = pd.to_datetime(tr.log_trace_time, unit='ms')
ek = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy(); del tr
v = ek.event_value.astype(str).str.replace('KEY:::', '', regex=False); key = v.str.split('---').str[0]
ek['printable'] = key.str.len().eq(1).values
EKT = {u: np.sort(x.t.values[x.printable.values]) for u, x in ek.groupby('user_id')}
KEYU = sorted(set(EKT) & set(users))

def propagate(enc, tau):
    """Entry mode of every token of the last snapshot: size of the inserted or replaced segment that brought it in."""
    prev = np.zeros(0, np.int32); lab = np.zeros(0, np.int8); tin = np.zeros(0, np.int64)  # lab: 1 typed, 2 inserted
    for k, cur in enumerate(enc):
        L = len(cur)
        if len(prev) == 0:
            lab = np.full(L, 1 if L <= tau else 2, np.int8); tin = np.full(L, k, np.int64); prev = cur; continue
        m = min(len(prev), L); ne = np.flatnonzero(prev[:m] != cur[:m]); pre = int(ne[0]) if len(ne) else m
        mm = min(len(prev), L) - pre; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
        a, b = prev[pre:len(prev) - suf], cur[pre:L - suf]
        if len(a) == 0 and len(b) == 0: prev = cur; continue
        nl = np.empty(L, np.int8); nt = np.empty(L, np.int64)
        nl[:pre] = lab[:pre]; nt[:pre] = tin[:pre]
        if suf: nl[L - suf:] = lab[len(prev) - suf:]; nt[L - suf:] = tin[len(prev) - suf:]
        if len(b):
            if len(a) == 0:
                ops = [('insert', 0, 0, 0, len(b))]
            else:
                ops = SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes()
            for op, i1, i2, j1, j2 in ops:
                if op == 'equal': nl[pre + j1:pre + j2] = lab[pre + i1:pre + i2]; nt[pre + j1:pre + j2] = tin[pre + i1:pre + i2]
                elif op in ('insert', 'replace'):
                    nl[pre + j1:pre + j2] = 1 if (j2 - j1) <= tau else 2; nt[pre + j1:pre + j2] = k
        lab, tin, prev = nl, nt, cur
    return prev, lab, tin


# ======================= R11 body =======================
users = users_all
tr = pd.read_pickle(f'{D}/trace.pkl'); tr['t'] = pd.to_datetime(tr.log_trace_time, unit='ms')
ek = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy()
ent = tr[(tr.learning_action_label == 'ASK_MULTI_AGENTS') & tr.event_value.astype(str).str.contains('Enter')][['user_id', 't']].copy()
del tr
v = ek.event_value.astype(str).str.replace('KEY:::', '', regex=False); key = v.str.split('---').str[0]
ek['printable'] = key.str.len().eq(1).values
EKT = {u: np.sort(x.t.values[x.printable.values]) for u, x in ek.groupby('user_id')}
KEYU = sorted(set(EKT) & set(users)); PAU = sorted(set(pa.user_id) & set(users))
REPT = {u: np.sort(pd.to_datetime(x.chatbot_utterance_time).values) for u, x in d.groupby('user_id')}

TK = {}; o = {}; F3 = Counter(); F4 = Counter(); xs_q = []
for kk, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in (g_.writing_content if g_ is not None else [])]
    F = P[u]; Fi = np.array([vocab.setdefault(x, len(vocab)) for x in F], np.int32)
    tbot, tpr = dialogue(u, 8); FG = grams(F, 8); FGs = {g: tuple(Fi[i:i + 8].tolist()) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), 8, set(FGs.values()))
    cls = classify(FGs, tbot, tpr, sc, 8); lab = project(F, 8, cls)
    # F-3 shared n-grams in an earlier reply; F-4 order margins of chatbot n-grams
    for g, c in cls.items():
        te = sc['ted'].get(FGs[g])
        if c == 'S':
            F3['S'] += 1
            if g in tbot and te is not None and tbot[g] <= te: F3['S_reply_before_editor'] += 1
            if g in tbot: F3['S_in_any_reply'] += 1
        if c in 'CE' and g in tbot and te is not None:
            dt = (te - tbot[g]).total_seconds()
            for dl in (2, 5):
                if c == 'C' and 0 <= dt < dl: F4[f'C_to_E_if_delta_minus{dl}'] += 1
                if c == 'E' and not (g in tpr and tpr[g] < tbot[g]) and -dl <= dt < 0: F4[f'E_to_C_if_delta_plus{dl}'] += 1
            F4[c] += 1
    last, tl, ti = propagate(enc, 5) if enc else (np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.int64))
    te_ = np.full(len(F), 2, np.int8); tt = np.full(len(F), -1, np.int64)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, Fi.tolist(), last.tolist(), autojunk=False).get_opcodes():
        if op == 'equal': te_[i1:i2] = tl[j1:j2]; tt[i1:i2] = ti[j1:j2]
    var = np.where(np.isin(lab, ['C', 'E', 'S']), lab, np.where(te_ == 1, 'T', 'X'))
    cent = set(tt[(var == 'C') & (tt >= 0)].tolist())
    xwc = (var == 'X') & np.isin(tt, list(cent)) if cent else np.zeros(len(F), bool)
    TK[u] = dict(F=F, lab=lab, var=var, tt=tt, times=times, xwc=xwc)
    if times:
        T0, T1 = times[0], times[-1]; span_ = (T1 - T0).total_seconds() or 1
        et = np.array([(times[k] - T0).total_seconds() / span_ if k >= 0 else np.nan for k in tt])
        last24 = np.array([((T1 - times[k]).total_seconds() <= 86400) if k >= 0 else False for k in tt])
        for c in 'CTX':
            m = var == c
            if m.any(): xs_q.append((u, c, int(m.sum()), int((m & last24).sum()), np.histogram(et[m & ~np.isnan(et)], bins=[0, .25, .5, .75, 1.0001])[0].tolist()))
    if kk % 50 == 0: print(kk, flush=True)
del WG, w
tot = sum(len(TK[u]['F']) for u in users)
o['F1_shares_token_entry'] = {c: float(np.mean([(TK[u]['var'] == c).mean() for u in users])) for c in 'CESTX'}
o['F1_shares_ngram_entry'] = {c: float(np.mean([(TK[u]['lab'] == c).mean() for u in users])) for c in 'CESTX'}
o['F1_X_with_C_token_entry'] = {'share_of_X_tokens': sum(int(TK[u]['xwc'].sum()) for u in users) / sum(int((TK[u]['var'] == 'X').sum()) for u in users),
                                'mean_share_of_proposal': float(np.mean([TK[u]['xwc'].mean() for u in users]))}
o['F1_kappa_token_vs_ngram'] = None
from sklearn.metrics import cohen_kappa_score
a_ = np.concatenate([TK[u]['lab'] for u in users]); b_ = np.concatenate([TK[u]['var'] for u in users])
o['F1_kappa_token_vs_ngram'] = float(cohen_kappa_score(a_, b_)); o['F1_token_agreement'] = float((a_ == b_).mean())
rows = []
for u in users:
    for i, (t_, c_, c0) in enumerate(zip(TK[u]['F'], TK[u]['var'], TK[u]['lab'])): rows.append((u, i, t_, c_, c0))
pd.DataFrame(rows, columns=['user_id', 'token_index', 'token', 'class', 'class_ngram_entry']).to_csv('results/flora_token_classes.csv.gz', index=False, compression='gzip')
sc_ = pd.DataFrame([{'user_id': u, **{c: (TK[u]['var'] == c).mean() for c in 'CESTX'}, 'X_with_C': TK[u]['xwc'].mean(),
                     **{c + '_ngram_entry': (TK[u]['lab'] == c).mean() for c in 'TX'}} for u in users])
sc_.to_csv('results/flora_student_classes.csv', index=False, float_format='%.6f')
U = sc_.C.values; o['U_unchanged_check'] = {'median': float(np.median(U)), 'ge50': float(np.mean(U >= .5))}

# F-2 timing of classes
q = pd.DataFrame(xs_q, columns=['u', 'cls', 'n', 'last24', 'quart'])
o['F2_share_entering_last24h'] = {c: float(g.last24.sum() / g.n.sum()) for c, g in q.groupby('cls')}
o['F2_share_by_task_quartile'] = {c: (np.sum(np.array(g.quart.tolist()), 0) / np.sum(np.array(g.quart.tolist()))).tolist() for c, g in q.groupby('cls')}

# F-2 logged pastes by class composition (68 students)
PR = []
for u in PAU:
    x = pa[pa.user_id == u]; F = TK[u]['F']; var = TK[u]['var']; Fg = {}
    for i, g in enumerate(grams(F, 8)): Fg.setdefault(g, []).append(i)
    T1 = TK[u]['times'][-1] if TK[u]['times'] else None; rt = REPT.get(u, np.array([], 'datetime64[ns]'))
    for t_, s_ in zip(x.t, x.txt):
        pt = tok(s_)
        if len(pt) < 8: continue
        cov = set()
        for g in grams(pt, 8):
            for i in Fg.get(g, []): cov.update(range(i, i + 8))
        if len(cov) < 8: continue
        cc = Counter(var[i] for i in cov); n_ = len(cov); shC, shX = cc['C'] / n_, cc['X'] / n_
        grp = 'C' if shC >= .8 else ('X' if shX >= .8 else ('mixed_CX' if shC >= .1 and shX >= .1 and shC + shX >= .8 else 'other'))
        prev_r = rt[rt <= np.datetime64(t_)]; since = (np.datetime64(t_) - prev_r[-1]) / np.timedelta64(1, 's') if len(prev_r) else np.nan
        PR.append(dict(u=u, grp=grp, len=len(pt), surv=n_, h_before_end=(T1 - t_).total_seconds() / 3600 if T1 is not None else np.nan, s_since_reply=since))
PR = pd.DataFrame(PR)
o['F2_pastes'] = {grp: {'n': int(len(g)), 'students': int(g.u.nunique()), 'len_median': float(g.len.median()), 'len_iqr': [float(g.len.quantile(.25)), float(g.len.quantile(.75))],
                        'hours_before_end_median': float(g.h_before_end.median()), 'share_within_24h_of_end': float((g.h_before_end <= 24).mean()),
                        's_since_reply_median': float(g.s_since_reply.median()), 'share_within_60s_of_reply': float((g.s_since_reply <= 60).mean()),
                        'share_within_10min_of_reply': float((g.s_since_reply <= 600).mean())} for grp, g in PR.groupby('grp')}
from scipy.stats import mannwhitneyu
cx = PR[PR.grp.isin(['C', 'X'])]
o['F2_pastes_C_vs_X_tests'] = {k: float(mannwhitneyu(cx[cx.grp == 'C'][k].dropna(), cx[cx.grp == 'X'][k].dropna()).pvalue) for k in ('len', 'h_before_end', 's_since_reply')}
o['F3_shared'] = dict(F3)
# F-4 clock direction
diffs = []
for u, x in d.groupby('user_id'):
    ee = np.sort(ent[ent.user_id == u].t.values)
    if not len(ee): continue
    for a in pd.to_datetime(x.user_ask_time).values:
        dd = (a - ee) / np.timedelta64(1, 's'); dd = dd[(dd >= 0) & (dd < 3600)]
        if len(dd): diffs.append(dd.min())
diffs = np.array(diffs); w30 = diffs[diffs < 30]
o['F4_clock'] = {'n_asks_with_preceding_enter_within_1h': int(len(diffs)), 'n_within_30s': int(len(w30)),
                 'ask_minus_enter_s_quartiles_within_30s': np.percentile(w30, [25, 50, 75]).tolist(), 'order_margins': dict(F4)}
# F-5 clustering of the keystroke audit
def icc1(df):
    g = df.groupby('u').agg(['mean', 'count'])['agree']; k = len(g); N_ = g['count'].sum(); n0 = (N_ - (g['count'] ** 2).sum() / N_) / (k - 1)
    gm = df.agree.mean(); msb = (g['count'] * (g['mean'] - gm) ** 2).sum() / (k - 1)
    msw = ((df.agree - df.u.map(g['mean'])) ** 2).sum() / (N_ - k) if N_ > k else np.nan
    return float((msb - msw) / (msb + (n0 - 1) * msw)) if (msb + (n0 - 1) * msw) > 0 else np.nan, float(N_ / k)
au = pd.read_csv('results/r9_audit_sample.csv'); crit = {'C': 'not_typed', 'E': 'student_side_evidence', 'T': 'typed', 'X': 'not_typed'}
o['F5_audit_clustering_r9'] = {}
for c, x in au.groupby('cls'):
    df = pd.DataFrame({'u': x.u, 'agree': x[crit[c]].astype(float)}); ic, m_ = icc1(df)
    o['F5_audit_clustering_r9'][c] = {'passages': int(len(df)), 'students': int(df.u.nunique()), 'mean_per_student': m_, 'icc': ic, 'design_effect': 1 + (m_ - 1) * (ic if ic and ic > 0 else 0)}
rngA = np.random.default_rng(2091); AUD = []
def audit(version):
    runs = []
    for u in KEYU:
        L = TK[u][version]; i = 0
        while i < len(L):
            j = i
            while j < len(L) and L[j] == L[i]: j += 1
            if j - i >= 8 and L[i] in 'TX': runs.append((u, L[i], i, j))
            i = j
    runs = pd.DataFrame(runs, columns=['u', 'cls', 'i', 'j']); res = {}
    for c in 'TX':
        x = runs[runs.cls == c].sample(frac=1, random_state=2091).groupby('u').head(3).head(50); ag = []
        for r in x.itertuples():
            d_ = TK[r.u]; ks = d_['tt'][r.i:r.j]; ks = ks[ks >= 0]
            if not len(ks): ag.append(c == 'X'); continue
            t1 = np.datetime64(d_['times'][ks.min()]) - np.timedelta64(3, 's'); t2 = np.datetime64(d_['times'][ks.max()]) + np.timedelta64(1, 's')
            if ks.min() > 0: t1 = min(t1, np.datetime64(d_['times'][ks.min() - 1]))
            ekt = EKT.get(r.u, np.array([], 'datetime64[ns]')); keys = int(((ekt >= t1) & (ekt <= t2)).sum())
            chars = len(' '.join(d_['F'][r.i:r.j]))
            ag.append(keys >= 0.5 * chars if c == 'T' else keys < 0.2 * chars)
        x = x.assign(agree=np.array(ag, float)); us_ = x.u.unique(); bs = []
        for _ in range(NB):
            b = rngA.choice(us_, len(us_)); bs.append(pd.concat([x[x.u == q_] for q_ in b]).agree.mean())
        ic, m_ = icc1(x[['u', 'agree']])
        res[c] = {'passages': int(len(x)), 'students': int(len(us_)), 'agree': float(x.agree.mean()), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  'icc': ic, 'mean_per_student': m_}
    return res
o['F5_audit_token_entry'] = audit('var'); o['F5_audit_ngram_entry'] = audit('lab')
json.dump(o, open('results/r11_flora.json', 'w'), indent=1, default=float); print(json.dumps({k: v for k, v in o.items()}, indent=1, default=float)[:6000])
