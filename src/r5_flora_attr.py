"""Plan R5, FLoRA attribution checks (ANALYSIS_LOG_R5.md F1-F6): within-document moves in X, segment-local entry rule,
students without logged pastes, token-level projections, clocks and ties, origin of E, timing baseline.
Usage: python -I src/r5_flora_attr.py DATA_DIR -> results/r5_flora_attr.json, results/_r5_flora_lab.pkl (local)"""
import sys, re, json, pickle, warnings
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2050); NB = 2000; J = 5; n = 8
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k=8: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
an = pd.read_pickle(f'{D}/annot.pkl'); tr = pd.read_pickle(f'{D}/trace.pkl')
pa = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy(); pa['t'] = pd.to_datetime(pa.log_trace_time, unit='ms')
ALL = sorted(set(d.user_id) & set(p.index)); P = {u: tok(p.loc[u, 'final_proposal']) for u in ALL}
H = set(g for t in an.highlight_text.dropna() for g in grams(tok(t))); Cc = Counter(g for u in ALL for g in set(grams(P[u])))
users = ALL[:int(sys.argv[2])] if len(sys.argv) > 2 else ALL
ORD = {'R4': 'CESTX', 'EC': 'ECSTX'}
THR = (0.2, 0.5, 0.8)

def scan(u):
    """One pass over all snapshots. Returns entry info for A-grams and transition statistics."""
    FG = grams(P[u]); need = set(FG); vocab = {}
    enc = lambda tk: [vocab.setdefault(x, len(vocab)) for x in tk]
    ted, mw, ms, prior = {}, {}, {}, {}
    seen = set(); prev, prevtok = [], []; ins_hist = []; gaps = []; tprev = None; nlarge = 0
    for t, c in zip(w[w.user_id == u].t, w[w.user_id == u].writing_content):
        tk = tok(c); ek = enc(tk)
        ops = SequenceMatcher(None, prev, ek, autojunk=False).get_opcodes()
        segs = [(j1, j2) for op, i1, i2, j1, j2 in ops if op in ('insert', 'replace')]
        dels = [(j1, j1) for op, i1, i2, j1, j2 in ops if op == 'delete']
        ins = sum(j2 - j1 for j1, j2 in segs); ins_hist.append(ins); nlarge += ins > 20
        if tprev is not None: gaps.append((t - tprev).total_seconds())
        tprev = t
        win = segs + dels; cand = set()
        for j1, j2 in win:
            for s in range(max(0, j1 - n + 1), min(len(tk) - n + 1, max(j2, j1 + 1))):
                cand.add(tuple(tk[s:s + n]))
        if not prev:
            cand = set(grams(tk))
        newA = [g for g in cand if g in need] if need else []
        if newA:
            # segment-local size and prior presence of the segment containing each new gram
            seginfo = []
            for j1, j2 in segs:
                inside = [tuple(tk[s:s + n]) for s in range(j1, j2 - n + 1)]
                pr = (sum(g in seen for g in inside) / len(inside)) if inside else 0.0
                seginfo.append((j1, j2, j2 - j1, pr))
            pos = {}
            for s in range(len(tk) - n + 1):
                g = tuple(tk[s:s + n])
                if g in need and g not in pos: pos[g] = s
            for g in newA:
                s = pos.get(g, 0); best = (0, 0.0, 0)
                for j1, j2, size, pr in seginfo:
                    ov = max(0, min(s + n, j2) - max(s, j1))
                    if ov > best[2]: best = (size, pr, ov)
                ted[g] = t; mw[g] = 'T' if ins <= J else 'X'; ms[g] = 'T' if best[0] <= J else 'X'; prior[g] = best[1]
                need.discard(g)
        # update seen with grams of this snapshot near edits (and all grams of the first snapshot)
        if not prev: seen |= set(grams(tk))
        else: seen |= cand
        prev = ek
    return dict(ted=ted, mw=mw, ms=ms, prior=prior, ins_hist=ins_hist, gaps=gaps, nlarge=nlarge,
                last=w[w.user_id == u].t.max() if (w.user_id == u).any() else pd.NaT, first=w[w.user_id == u].t.min() if (w.user_id == u).any() else pd.NaT)

def label(u, sc, mode_key='mw', moved_thr=None):
    x = d[d.user_id == u]; FG = grams(P[u]); tbot, tpr, plen = {}, {}, {}
    for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(s)): tbot.setdefault(g, pd.Timestamp(t))
    for t, s in zip(x.user_ask_time, x.user_utterance_text):
        tt = tok(s)
        for g in grams(tt):
            if g not in tpr: tpr[g] = pd.Timestamp(t); plen[g] = len(tt)
    ted, mode = sc['ted'], sc[mode_key]; cls = {}
    for g in set(FG):
        bot = g in tbot; echo = bot and g in tpr and tpr[g] < tbot[g]; shared = g in H or Cc[g] - 1 >= 5
        if bot and not echo and not shared and g in ted and tbot[g] <= ted[g]: cls[g] = 'C'
        elif bot and (echo or (g in ted and ted[g] < tbot[g])): cls[g] = 'E'
        elif shared: cls[g] = 'S'
        else:
            m = mode.get(g, 'X')
            if m == 'X' and moved_thr is not None and g in ted and sc['prior'].get(g, 0) >= moved_thr: m = 'M'
            cls[g] = m
    return FG, cls, tbot, tpr, plen

def project(F, FG, cls, order):
    rank = {c: i for i, c in enumerate(order)}; best = np.full(len(F), 99)
    for i, g in enumerate(FG): best[i:i + n] = np.minimum(best[i:i + n], rank[cls[g]])
    return np.array([order[b] if b < 99 else 'X' for b in best])

def majority(F, FG, cls, order='CESTX'):
    votes = [Counter() for _ in F]
    for i, g in enumerate(FG):
        for j in range(i, i + n): votes[j][cls[g]] += 1
    rank = {c: i for i, c in enumerate(order)}
    return np.array([min(v, key=lambda c: (-v[c], rank[c])) if v else 'X' for v in votes])

SC, LAB = {}, {}
for k, u in enumerate(users):
    SC[u] = scan(u)
    if k % 25 == 0: print('scan', k, flush=True)
out = {}
U1 = np.load('results/U_R1.npy')[:len(users)]
bci = lambda f, v: [float(np.percentile([f(rng.choice(v, len(v))) for _ in range(NB)], q)) for q in (2.5, 97.5)]
shares = {}
for u in users:
    F = P[u]; FG, cls, tbot, tpr, plen = label(u, SC[u])
    _, cls_s, *_ = label(u, SC[u], 'ms')
    L = {'R4': project(F, FG, cls, 'CESTX'), 'EC': project(F, FG, cls, 'ECSTX'), 'MAJ': majority(F, FG, cls), 'SEG': project(F, FG, cls_s, 'CESTX')}
    for th in THR:
        _, cls_m, *_ = label(u, SC[u], 'mw', th); L[f'M{th}'] = project(F, FG, cls_m, 'CESTMX')
    LAB[u] = dict(L=L, cls=cls, tbot=tbot, tpr=tpr, plen=plen, FG=FG)
# check: R4 projection reproduces committed classes
out['check_C_vs_Ustar_maxdiff'] = float(np.abs(np.array([(LAB[u]['L']['R4'] == 'C').mean() for u in users]) - U1).max())
def table(key, letters='CESTX'):
    sh = pd.DataFrame({c: [(LAB[u]['L'][key] == c).mean() for u in users] for c in letters}, index=users)
    return sh, {c: {'mean': float(sh[c].mean()), 'median': float(sh[c].median()), 'mean_ci': bci(np.mean, sh[c].values)} for c in letters}
sh_r4, out['classes_R4'] = table('R4')
for key in ['EC', 'MAJ', 'SEG']:
    sh, out[f'classes_{key}'] = table(key)
    ch = sum(int((LAB[u]['L'][key] != LAB[u]['L']['R4']).sum()) for u in users); tot = sum(len(P[u]) for u in users)
    out[f'classes_{key}']['tokens_changed_share'] = ch / tot
    Uk = sh['C'].values
    out[f'classes_{key}']['U_median'] = float(np.median(Uk)); out[f'classes_{key}']['ge50'] = float((Uk >= .5).mean()); out[f'classes_{key}']['ge25'] = float((Uk >= .25).mean())
    out[f'classes_{key}']['spearman_with_Ustar'] = float(pd.Series(Uk).rank().corr(pd.Series(U1).rank()))
# F1 moved text
for th in THR:
    sh, o = table(f'M{th}', 'CESTMX'); o['M_share_of_X_R4'] = float(sum(int((LAB[u]['L'][f'M{th}'] == 'M').sum()) for u in users) / max(1, sum(int((LAB[u]['L']['R4'] == 'X').sum()) for u in users)))
    o['X_median_after'] = float(sh['X'].median()); out[f'moved_{th}'] = o
# gram-level prior presence distribution for X and T grams
pri_X = [SC[u]['prior'][g] for u in users for g, c in LAB[u]['cls'].items() if c == 'X' and g in SC[u]['prior']]
pri_T = [SC[u]['prior'][g] for u in users for g, c in LAB[u]['cls'].items() if c == 'T' and g in SC[u]['prior']]
out['prior_presence'] = {'X_grams': len(pri_X), 'X_share_ge': {str(t): float(np.mean(np.array(pri_X) >= t)) for t in THR}, 'X_share_zero': float(np.mean(np.array(pri_X) == 0)),
                         'T_grams': len(pri_T), 'T_share_ge': {str(t): float(np.mean(np.array(pri_T) >= t)) for t in THR}}
# transitions
ih = np.concatenate([SC[u]['ins_hist'][1:] for u in users]); gp = np.concatenate([SC[u]['gaps'] for u in users])
out['transitions'] = {'n': int(len(ih)), 'ins_share_0': float((ih == 0).mean()), 'ins_share_le1': float((ih <= 1).mean()), 'ins_share_le5': float((ih <= 5).mean()),
                      'ins_share_gt20': float((ih > 20).mean()), 'ins_quantiles': {q: float(np.percentile(ih, q)) for q in (50, 90, 99, 99.9)},
                      'gap_s_quantiles': {q: float(np.percentile(gp, q)) for q in (10, 50, 90, 99)}, 'gap_share_le2s': float((gp <= 2).mean())}
# F2 students without logged pastes
haspaste = set(pa.user_id); firstpaste = pa.t.min(); lastpaste = pa.t.max()
grp = {}
for k, v in [('with', True), ('without', False)]:
    us = [u for u in users if (u in haspaste) == v]
    grp[k] = {'n': len(us), 'X_mean': float(sh_r4.loc[us, 'X'].mean()), 'large_insertions_median': float(np.median([SC[u]['nlarge'] for u in us])),
              'share_ending_before_first_logged_paste': float(np.mean([SC[u]['last'] < firstpaste for u in us])),
              'share_starting_after_last_logged_paste': float(np.mean([SC[u]['first'] > lastpaste for u in us])),
              'share_active_in_paste_window': float(np.mean([(SC[u]['last'] >= firstpaste) and (SC[u]['first'] <= lastpaste) for u in us])),
              'activity_start_median': str(pd.Series([SC[u]['first'] for u in us]).median()), 'activity_end_median': str(pd.Series([SC[u]['last'] for u in us]).median())}
grp['first_logged_paste'] = str(firstpaste); grp['last_logged_paste'] = str(lastpaste)
grp['pastes_by_day'] = {str(k): int(v) for k, v in pa.t.dt.date.value_counts().sort_index().items()}
grp['snapshots_by_day'] = {str(k): int(v) for k, v in w[w.user_id.isin(users)].t.dt.date.value_counts().sort_index().items()}
out['paste_logging'] = grp
# F4 ties for C grams
dt = np.array([(SC[u]['ted'][g] - LAB[u]['tbot'][g]).total_seconds() for u in users for g, c in LAB[u]['cls'].items() if c == 'C'])
out['C_ties'] = {'n_C_grams': int(len(dt)), 'share_within_10s': float((np.abs(dt) <= 10).mean()), 'share_within_60s': float((np.abs(dt) <= 60).mean()), 'median_s': float(np.median(dt))}
# F5 origin of E
eg = [(u, g) for u in users for g, c in LAB[u]['cls'].items() if c == 'E']
ed_large = [SC[u]['mw'][g] == 'X' for u, g in eg if g in SC[u]['ted']]
pf = [(u, g) for u, g in eg if g in LAB[u]['tpr'] and g in LAB[u]['tbot'] and LAB[u]['tpr'][g] < LAB[u]['tbot'][g]]
out['E_origin'] = {'E_grams': len(eg), 'editor_entry_large_share': float(np.mean(ed_large)), 'prompt_first_share': len(pf) / max(1, len(eg)),
                   'prompt_first_long_prompt_share': float(np.mean([LAB[u]['plen'][g] >= 50 for u, g in pf]))}
# F6 timing by class (token entry = earliest editor time among covering grams of the token's own class)
lastall = max(SC[u]['last'] for u in users if pd.notna(SC[u]['last']))
tim = {c: {'tot': 0, 'h24': 0, 'h2': 0, 'dl24': 0, 'dl72': 0} for c in 'CETX'}
for u in users:
    L = LAB[u]['L']['R4']; FG = LAB[u]['FG']; cls = LAB[u]['cls']; ted = SC[u]['ted']; last = SC[u]['last']
    tmin = [None] * len(P[u])
    for i, g in enumerate(FG):
        if g in ted:
            for j in range(i, i + n):
                if cls[g] == L[j] and (tmin[j] is None or ted[g] < tmin[j]): tmin[j] = ted[g]
    for j, t in enumerate(tmin):
        c = L[j]
        if t is None or c not in tim: continue
        h = (last - t).total_seconds() / 3600; hd = (lastall - t).total_seconds() / 3600
        tim[c]['tot'] += 1; tim[c]['h24'] += h <= 24; tim[c]['h2'] += h <= 2; tim[c]['dl24'] += hd <= 24; tim[c]['dl72'] += hd <= 72
out['timing'] = {c: {k: (v[k] / max(v['tot'], 1) if k != 'tot' else v['tot']) for k in v} for c, v in tim.items()}
out['timing']['deadline_proxy'] = str(lastall)
out['timing']['last_snapshot_hours_before_proxy'] = {q: float(np.percentile([(lastall - SC[u]['last']).total_seconds() / 3600 for u in users], q)) for q in (10, 50, 90)}
sh_r4.to_csv('results/r5_flora_classes.csv')
pickle.dump({u: {k: LAB[u]['L'][k] for k in LAB[u]['L']} for u in users}, open('results/_r5_flora_lab.pkl', 'wb'))
json.dump(out, open('results/r5_flora_attr.json', 'w'), indent=1); print(json.dumps(out, indent=1))
