"""Analysis log R9, FLoRA (ANALYSIS_LOG_R9.md F-1 to F-7) and an audit of sampled passages against keystroke and
selection events that the attribution does not use. Readers, scan and rules are copied from src/r8_flora.py.
Usage: python -I src/r9_flora.py DATA_DIR [R2_PASSAGES_PKL] -> results/r9_flora.json, results/r9_audit_sample.csv"""
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



# ======================= R9 body =======================
import statsmodels.formula.api as smf, pickle
PASS = sys.argv[2] if len(sys.argv) > 2 else None
users = users_all
tr = pd.read_pickle(f'{D}/trace.pkl'); tr['t'] = pd.to_datetime(tr.log_trace_time, unit='ms')
ek = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy()
ck = tr[(tr.learning_action_label == 'ASK_MULTI_AGENTS') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy()
sel = tr[(tr.learning_action_label == 'READ_FEEDBACK_MULTI_AGENTS') & tr.event_value.astype(str).str.startswith('MOUSE_SELECT:::')][['user_id', 't', 'event_value']].copy()
del tr
def keyinfo(df):
    df = df.sort_values(['user_id', 't']).copy(); v = df.event_value.astype(str).str.replace('KEY:::', '', regex=False)
    key = v.str.split('---').str[0]
    df['printable'] = key.str.len().eq(1).values
    df['mod'] = key.isin(['Control', 'Meta']).values
    df['isv'] = v.str.endswith('KeyV').values
    return df
ek, ck = keyinfo(ek), keyinfo(ck)
def pastes_shortcut(df):  # time of each Control/Meta + V (V within 1.5 s after a modifier)
    out = {}
    for u, x in df.groupby('user_id'):
        tm = x.t.values; mod = x['mod'].values; isv = x.isv.values; res = []
        lastmod = None
        for t_, m_, v_ in zip(tm, mod, isv):
            if m_: lastmod = t_
            elif v_ and lastmod is not None and (t_ - lastmod) / np.timedelta64(1, 's') <= 1.5: res.append(t_)
        out[u] = np.array(res, dtype='datetime64[ns]')
    return out
EPS, CPS = pastes_shortcut(ek), pastes_shortcut(ck)
EKT = {u: x.t.values[x.printable.values] for u, x in ek.groupby('user_id')}
CKT = {u: x.t.values[x.printable.values] for u, x in ck.groupby('user_id')}
SEL = {u: [(t_, set(grams(tok(s.replace('MOUSE_SELECT:::', '')), 8))) for t_, s in zip(x.t, x.event_value.astype(str))] for u, x in sel.groupby('user_id')}
KEYU = sorted(set(EKT) & set(users)); print('keystroke students', len(KEYU), flush=True)
PT = {u: np.sort(pa[pa.user_id == u].t.values) for u in users}; PASTEU = [u for u in users if len(PT[u])]

def classify_full(FGs, tbot, tpr, sc, n, tau=5):
    cls, route = {}, {}
    for g, gi in FGs.items():
        bot = g in tbot; pf = bot and g in tpr and tpr[g] < tbot[g]; shared = g in H[n] or Cc[n][g] - 1 >= 5
        te = sc['ted'].get(gi)
        if bot and not pf and not shared and te is not None and tbot[g] <= te: cls[g] = 'C'
        elif bot and (pf or (te is not None and te < tbot[g])): cls[g] = 'E'; route[g] = 'prompt' if pf else 'editor'
        elif shared: cls[g] = 'S'
        elif te is None: cls[g] = 'X'
        else: cls[g] = 'T' if sc['ins'][gi] <= tau else 'X'
    return cls, route

THIN = (5, 15, 60, 300); TAUS = (1, 2, 3, 5, 10, 20, 50)
ST = {}; RUNT = []; tot_snap = 0
for k, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; contents = list(g_.writing_content) if g_ is not None else []
    t0 = time.perf_counter()
    vocab = {}; enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in contents]
    F = P[u]; Fi = [vocab.setdefault(x, len(vocab)) for x in F]
    tbot, tpr = dialogue(u, 8); FG = grams(F, 8); FGs = {g: tuple(Fi[i:i + 8]) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), 8, set(FGs.values()))
    cls, route = classify_full(FGs, tbot, tpr, sc, 8); lab = project(F, 8, cls)
    RUNT.append((u, len(enc), len(F), time.perf_counter() - t0)); tot_snap += len(enc)
    rec = {'F': F, 'FG': FG, 'FGs': FGs, 'cls': cls, 'route': route, 'lab': lab, 'tbot': tbot, 'tpr': tpr,
           'ted': {g: sc['ted'].get(FGs[g]) for g in FG}, 'ins': {g: sc['ins'].get(FGs[g]) for g in FG}}
    # thinning: tokens changing class (all classes)
    th = {}
    for s_ in THIN:
        idx, last = [], None
        for i, t_ in enumerate(times):
            if last is None or (t_ - last).total_seconds() >= s_: idx.append(i); last = t_
        if times and (not idx or idx[-1] != len(times) - 1): idx.append(len(times) - 1)
        sct = scan(enc, times, idx, 8, set(FGs.values()))
        for tau in (5, 20):
            c2, _ = classify_full(FGs, tbot, tpr, sct, 8, tau); th[(s_, tau)] = project(F, 8, c2)
    rec['thin'] = th
    rec['tau'] = {tau: project(F, 8, classify_full(FGs, tbot, tpr, sc, 8, tau)[0]) for tau in TAUS}
    rec['tau_grams'] = {tau: classify_full(FGs, tbot, tpr, sc, 8, tau)[0] for tau in TAUS} if u in PASTEU else None
    ST[u] = rec
    if k % 25 == 0: print(k, flush=True)
del WG, w
out = {'n_students': len(users)}
lab_ = lambda u: ST[u]['lab']; Us = np.array([(lab_(u) == 'C').mean() for u in users])
out['check_C_vs_U_R1'] = float(np.abs(Us - np.load('results/U_R1.npy')).max())

# ---- F-1 echo by entry mode and route
EX = {}; Erow = Counter(); ERT = {}
for u in users:
    r = ST[u]; F = r['F']; L = r['lab']; ex = np.zeros(len(F), bool); et = np.zeros(len(F), bool); pf = np.zeros(len(F), bool); ef = np.zeros(len(F), bool)
    for i, g in enumerate(r['FG']):
        if r['cls'][g] != 'E': continue
        large = r['ins'][g] is None or r['ins'][g] > 5
        Erow['grams'] += 1; Erow['large'] += large; Erow[r['route'][g]] += 1; Erow[r['route'][g] + '_large'] += large
        (ex if large else et)[i:i + 8] = True; (pf if r['route'][g] == 'prompt' else ef)[i:i + 8] = True
    isE = L == 'E'; EX[u] = isE & ex & ~et  # E-X: every covering E n-gram entered as a large insertion
    ERT[u] = {'E': isE.mean(), 'E_X': (isE & ex & ~et).mean(), 'E_T': (isE & et).mean(), 'E_prompt': (isE & pf).mean(), 'E_editor_only': (isE & ef & ~pf).mean()}
er = pd.DataFrame(ERT).T
out['F1_echo'] = {'gram_counts': dict(Erow), 'mean_shares': {c: float(er[c].mean()) for c in er.columns},
                  'students_with_any_EX': float((er.E_X > 0).mean()), 'n_students_any_EX': int((er.E_X > 0).sum()),
                  'Ustar_median_with_EX': float(np.median(Us[er.E_X.values > 0])), 'Ustar_median_without_EX': float(np.median(Us[er.E_X.values == 0])),
                  'ge50_C': float((Us >= .5).mean()), 'ge50_C_plus_EX': float(((Us + er.E_X.values) >= .5).mean()),
                  'ge50_C_plus_E': float(((Us + er.E.values) >= .5).mean()),
                  'median_C_plus_EX': float(np.median(Us + er.E_X.values))}
# ---- F-2 runtime
rt = pd.DataFrame(RUNT, columns=['u', 'snapshots', 'tokens', 'sec'])
out['F2_runtime'] = {'total_sec': float(rt.sec.sum()), 'median_sec_per_student': float(rt.sec.median()), 'p90_sec': float(rt.sec.quantile(.9)),
                     'max_sec': float(rt.sec.max()), 'snapshots_total': int(tot_snap), 'us_per_snapshot': float(rt.sec.sum() / max(tot_snap, 1) * 1e6),
                     'rho_sec_snapshots': float(stats.spearmanr(rt.sec, rt.snapshots)[0])}
# ---- F-3 clock-check group
grp = np.array([u in set(KEYU) for u in users]); score_ = p.loc[users, 'proposal_score'].values.astype(float)
X_ = np.array([(lab_(u) == 'X').mean() for u in users])
def slope(m):
    zz = (Us[m] - Us[m].mean()) / Us[m].std(); f = sm.OLS(score_[m], sm.add_constant(zz)).fit(cov_type='HC3')
    return {'per_sd': float(f.params[1]), 'ci': [float(c) for c in f.conf_int()[1]]}
out['F3_clock_group'] = {g: {'n': int(m.sum()), 'median_U': float(np.median(Us[m])), 'mean_U': float(Us[m].mean()), 'ge50': float((Us[m] >= .5).mean()),
                             'mean_X': float(X_[m].mean()), 'grade_slope': slope(m)} for g, m in (('keystrokes', grp), ('others', ~grp))}
# ---- F-4 thinning, all classes
tot = sum(len(ST[u]['F']) for u in users)
out['F4_thinning'] = {f'{s_}s_tau{tau}': {'tokens_changing_class': sum(int((ST[u]['thin'][(s_, tau)] != ST[u]['lab']).sum()) for u in users) / tot,
                                          'changing_into_or_out_of_C_E': sum(int(((ST[u]['thin'][(s_, tau)] != ST[u]['lab']) & (np.isin(ST[u]['thin'][(s_, tau)], ['C', 'E']) | np.isin(ST[u]['lab'], ['C', 'E']))).sum()) for u in users) / tot}
                      for s_ in THIN for tau in (5, 20)}
# ---- F-5 tau and paste window (68 students)
F5 = {}
for tau in TAUS:
    chg = sum(int(((ST[u]['tau'][tau] != ST[u]['tau'][5]) & np.isin(ST[u]['tau'][5], ['T', 'X'])).sum()) for u in users) / max(1, sum(int(np.isin(ST[u]['tau'][5], ['T', 'X']).sum()) for u in users))
    row = {'TX_tokens_changing_vs_tau5': chg}
    for win in ((5, 10, 30, 60) if tau == 5 else (10,)):
        cnt = Counter()
        for u in PASTEU:
            r = ST[u]; pt = PT[u]
            for g, c in r['tau_grams'][tau].items():
                if c not in 'TX' or r['ted'][g] is None: continue
                te = np.datetime64(r['ted'][g]); hit = bool(len(pt)) and np.min(np.abs((pt - te) / np.timedelta64(1, 's'))) <= win
                cnt[c] += 1; cnt[c + '_paste'] += hit
        row[f'win{win}'] = {'T_with_paste': cnt['T_paste'] / max(cnt['T'], 1), 'X_with_paste': cnt['X_paste'] / max(cnt['X'], 1), 'T_grams': cnt['T'], 'X_grams': cnt['X']}
    F5[tau] = row
out['F5_tau_window'] = F5
# ---- F-7 contribution ratio
Tsh = np.array([(lab_(u) == 'T').mean() for u in users]); Esh = er.E.values
out['F7_contribution_ratio'] = {'mean_T': float(Tsh.mean()), 'median_T': float(np.median(Tsh)), 'mean_T_plus_E': float((Tsh + Esh).mean()),
                                'rho_T_with_1_minus_U': float(stats.spearmanr(Tsh, 1 - Us)[0]), 'mean_1_minus_U': float((1 - Us).mean()),
                                'share_students_T_lt_10pct': float((Tsh < .1).mean()), 'share_students_1mU_gt_50pct_and_T_lt_10pct': float(((1 - Us) > .5).__and__(Tsh < .1).mean())}

# ---- AUDIT: independent keystroke and selection evidence for sampled passages (71 students)
rngA = np.random.default_rng(2091); runs = []
for u in KEYU:
    r = ST[u]; L = r['lab']; F = r['F']; i = 0
    while i < len(L):
        j = i
        while j < len(L) and L[j] == L[i]: j += 1
        if j - i >= 8 and L[i] in 'CETX': runs.append((u, L[i], i, j))
        i = j
runs = pd.DataFrame(runs, columns=['u', 'cls', 'i', 'j'])
samp = []
for c in 'CETX':
    x = runs[runs.cls == c].sample(frac=1, random_state=2091); x = x.groupby('u').head(3); samp.append(x.head(50))
samp = pd.concat(samp).reset_index(drop=True)
def evidence(row):
    u, c, i, j = row.u, row.cls, row.i, row.j; r = ST[u]; FG = r['FG']
    gs = [FG[k] for k in range(max(0, i - 7), min(len(FG), j - 7)) if FG[k] in r['ted'] and k + 8 <= j and k >= i]
    if not gs: gs = [FG[k] for k in range(max(0, i - 7), min(len(FG), j)) if r['ted'].get(FG[k]) is not None]
    tes = [r['ted'][g] for g in gs if r['ted'].get(g) is not None]
    if not tes: return {}
    t1, t2 = np.datetime64(min(tes)) - np.timedelta64(3, 's'), np.datetime64(max(tes)) + np.timedelta64(1, 's')
    text = ' '.join(r['F'][i:j]); chars = len(text)
    ekt = EKT.get(u, np.array([], 'datetime64[ns]')); keys = int(((ekt >= t1) & (ekt <= t2)).sum())
    eps = EPS.get(u, np.array([], 'datetime64[ns]')); pasted = bool(((eps >= t1 - np.timedelta64(5, 's')) & (eps <= t2)).sum())
    pg = set(FG[k] for k in range(i, j - 7)) if j - i >= 8 else set()
    selb = any(pd.Timestamp(ts) <= pd.Timestamp(t2) and (sg & pg) for ts, sg in SEL.get(u, []))
    ev = {'keys': keys, 'chars': chars, 'pasted_shortcut': pasted, 'selected_in_chat_before_entry': selb, 'typed': keys >= 0.5 * chars and not pasted,
          'not_typed': keys < 0.2 * chars}
    if c == 'E':
        rts = [r['route'].get(g) for g in pg if g in r['route']]; ev['route'] = 'prompt' if 'prompt' in rts else 'editor'
        if ev['route'] == 'prompt':
            tp = min(r['tpr'][g] for g in pg if g in r['tpr']); tp = np.datetime64(tp)
            ckt = CKT.get(u, np.array([], 'datetime64[ns]')); cps = CPS.get(u, np.array([], 'datetime64[ns]'))
            w0 = tp - np.timedelta64(600, 's'); ckeys = int(((ckt >= w0) & (ckt <= tp)).sum()); cpaste = bool(((cps >= w0) & (cps <= tp)).sum())
            ev.update({'chat_keys_before_ask': ckeys, 'chat_paste_before_ask': cpaste, 'student_side_evidence': cpaste or ckeys >= 0.5 * chars})
        else:
            tb = min(r['tbot'][g] for g in pg if g in r['tbot']); ev['student_side_evidence'] = bool(min(tes) < tb) and (ev['typed'] or pasted or keys > 0)
        ev['selected_before_student_side'] = selb
    return ev
E_ = [evidence(rw) for rw in samp.itertuples()]
samp = pd.concat([samp, pd.DataFrame(E_)], axis=1)
def boot(mask, val):
    x = samp[mask]; us = x.u.unique(); v = []
    for _ in range(NB):
        b = rngA.choice(us, len(us)); xx = pd.concat([x[x.u == q] for q in b]); v.append(xx[val].mean())
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
AU = {}
crit = {'T': 'typed', 'X': 'not_typed', 'C': 'not_typed', 'E': 'student_side_evidence'}
for c in 'CETX':
    m = samp.cls == c; x = samp[m]
    AU[c] = {'passages': int(m.sum()), 'students': int(x.u.nunique()), 'criterion': crit[c], 'agree': float(x[crit[c]].astype(float).mean()),
             'agree_ci': boot(m, crit[c]), 'typed': float(x.typed.mean()), 'not_typed': float(x.not_typed.mean()), 'pasted_shortcut': float(x.pasted_shortcut.mean()),
             'selected_in_chat_before_entry': float(x.selected_in_chat_before_entry.mean())}
    if c == 'E': AU[c]['route_prompt_share'] = float((x.route == 'prompt').mean()); AU[c]['selected_before_student_side'] = float(x.selected_before_student_side.mean())
out['AUDIT'] = AU
samp.drop(columns=[]).to_csv('results/r9_audit_sample.csv', index=False)
# ---- F-6 paragraph length by detector class
if PASS:
    Rr = pd.read_pickle(PASS); Rr = Rr[Rr.user_id.isin(users)].copy()
    def pcl(u, text):
        t = tok(text); F = ST[u]['F']; n_ = len(t)
        for i in range(len(F) - n_ + 1):
            if F[i:i + n_] == t: return ST[u]['lab'][i:i + n_]
    pc = [pcl(u, s) for u, s in zip(Rr.user_id, Rr.text)]
    for c in 'TX': Rr[c] = [float((a == c).mean()) if a is not None else np.nan for a in pc]
    Rr['len'] = [len(tok(s)) for s in Rr.text]
    Rr['cls3'] = np.where(Rr.label == 'chatbot', 'chatbot', np.where((Rr.label == 'composed') & (Rr['T'] >= .9), 'typed', np.where((Rr.label == 'composed') & (Rr['X'] >= .9), 'untraced', 'other')))
    out['F6_paragraph_length'] = {c: {'n': int((Rr.cls3 == c).sum()), 'median_tokens': float(Rr[Rr.cls3 == c].len.median()), 'iqr': [float(Rr[Rr.cls3 == c].len.quantile(.25)), float(Rr[Rr.cls3 == c].len.quantile(.75))]} for c in ('chatbot', 'typed', 'untraced')}
json.dump(out, open('results/r9_flora.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
