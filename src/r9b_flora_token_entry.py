"""Analysis log R9, F-8: token-level entry mode. Each token of the last snapshot carries the size of the edit that
inserted it (propagated through every snapshot transition); T/X for tokens outside C, E and S is then decided per token
instead of per n-gram. Keystroke support of T and X runs and of typed detector paragraphs under both versions.
Usage: python -I src/r9b_flora_token_entry.py DATA_DIR [R2_PASSAGES_PKL] -> results/r9b_flora_token_entry.json"""
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
PASS = sys.argv[2] if len(sys.argv) > 2 else None
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

OUT = {}; TK = {}; rows = []
for kk, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in (g_.writing_content if g_ is not None else [])]
    F = P[u]; Fi = np.array([vocab.setdefault(x, len(vocab)) for x in F], np.int32)
    tbot, tpr = dialogue(u, 8); FG = grams(F, 8); FGs = {g: tuple(Fi[i:i + 8].tolist()) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), 8, set(FGs.values()))
    cls = classify(FGs, tbot, tpr, sc, 8); lab = project(F, 8, cls)
    last, tl, ti = propagate(enc, 5) if enc else (np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.int64))
    te = np.full(len(F), 2, np.int8); tt = np.full(len(F), -1, np.int64)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, Fi.tolist(), last.tolist(), autojunk=False).get_opcodes():
        if op == 'equal': te[i1:i2] = tl[j1:j2]; tt[i1:i2] = ti[j1:j2]
    var = np.where(np.isin(lab, ['C', 'E', 'S']), lab, np.where(te == 1, 'T', 'X'))
    TK[u] = dict(F=F, lab=lab, var=var, tt=tt, times=times)
    if kk % 50 == 0: print(kk, flush=True)
del WG, w
tot = sum(len(TK[u]['F']) for u in users)
o = {'shares_primary': {c: float(np.mean([(TK[u]['lab'] == c).mean() for u in users])) for c in 'CESTX'},
     'shares_token_entry': {c: float(np.mean([(TK[u]['var'] == c).mean() for u in users])) for c in 'CESTX'},
     'tokens_T_to_X': sum(int(((TK[u]['lab'] == 'T') & (TK[u]['var'] == 'X')).sum()) for u in users) / tot,
     'tokens_X_to_T': sum(int(((TK[u]['lab'] == 'X') & (TK[u]['var'] == 'T')).sum()) for u in users) / tot,
     'share_of_T_moving_to_X': sum(int(((TK[u]['lab'] == 'T') & (TK[u]['var'] == 'X')).sum()) for u in users) / sum(int((TK[u]['lab'] == 'T').sum()) for u in users)}
# keystroke support per labelled run (both versions), same sampling and criteria as the R9 audit
rngA = np.random.default_rng(2091)
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
            # previous snapshot before the earliest entry, to include keys typed just before the snapshot was saved
            if ks.min() > 0: t1 = min(t1, np.datetime64(d_['times'][ks.min() - 1]))
            ekt = EKT.get(r.u, np.array([], 'datetime64[ns]')); keys = int(((ekt >= t1) & (ekt <= t2)).sum())
            chars = len(' '.join(d_['F'][r.i:r.j]))
            ag.append(keys >= 0.5 * chars if c == 'T' else keys < 0.2 * chars)
        x = x.assign(agree=ag); us = x.u.unique(); bs = []
        for _ in range(NB):
            b = rngA.choice(us, len(us)); bs.append(pd.concat([x[x.u == q] for q in b]).agree.mean())
        res[c] = {'passages': int(len(x)), 'students': int(len(us)), 'agree': float(x.agree.mean()), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return res
o['audit_primary'] = audit('lab'); o['audit_token_entry'] = audit('var')
# detector paragraphs: typed (>= 90% T) under each version, and keystroke support of typed paragraphs in the keystroke group
if PASS:
    Rr = pd.read_pickle(PASS); Rr = Rr[(Rr.user_id.isin(users)) & (Rr.label == 'composed')].copy()
    def span(u, text):
        t = tok(text); F = TK[u]['F']; n_ = len(t)
        for i in range(len(F) - n_ + 1):
            if F[i:i + n_] == t: return i, i + n_
    sp = [span(u, s) for u, s in zip(Rr.user_id, Rr.text)]
    for ver in ('lab', 'var'):
        Rr['T_' + ver] = [float((TK[u][ver][a:b] == 'T').mean()) if s is not None else np.nan for u, s, (a, b) in zip(Rr.user_id, sp, [s if s else (0, 0) for s in sp])]
    o['typed_paragraphs'] = {'primary': int((Rr.T_lab >= .9).sum()), 'token_entry': int((Rr.T_var >= .9).sum()),
                             'both': int(((Rr.T_lab >= .9) & (Rr.T_var >= .9)).sum())}
    ks = []
    for (u, s_), tl_ in zip(zip(Rr.user_id, sp), Rr.T_lab):
        if s_ is None or tl_ < .9 or u not in EKT: continue
        d_ = TK[u]; tt_ = d_['tt'][s_[0]:s_[1]]; tt_ = tt_[tt_ >= 0]
        if not len(tt_): continue
        t1 = np.datetime64(d_['times'][max(tt_.min() - 1, 0)]); t2 = np.datetime64(d_['times'][tt_.max()]) + np.timedelta64(1, 's')
        keys = int(((EKT[u] >= t1) & (EKT[u] <= t2)).sum()); ks.append(keys / len(' '.join(d_['F'][s_[0]:s_[1]])))
    o['typed_paragraphs_keystroke_group'] = {'n': len(ks), 'share_keys_ge_half_chars': float(np.mean(np.array(ks) >= .5)) if ks else None,
                                             'median_keys_per_char': float(np.median(ks)) if ks else None}
json.dump(o, open('results/r9b_flora_token_entry.json', 'w'), indent=1); print(json.dumps(o, indent=1))
