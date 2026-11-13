"""Analysis log R10, O-1 and O-2: online attribution on the study logs; agreement with the offline labels, latency
per snapshot, peak memory and run time per student and per cohort.
Usage: python -I src/r10_online.py DATA_DIR RS_DIR -> results/r10_online.json"""
import sys, os, json, time, tracemalloc, platform, re
from collections import Counter
from difflib import SequenceMatcher
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demo'))
from provenance import tokens, ngrams, ORDER
from provenance_online import OnlineAttributor
D, RSD = sys.argv[1], sys.argv[2]

def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b); po = (a == b).mean(); pe = sum((a == c).mean() * (b == c).mean() for c in ORDER)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0

def run_online(events, shared, final, n, tau):
    att = OnlineAttributor(shared=shared, n=n, tau=tau); lat = []
    for t, k, x in sorted(events, key=lambda e: (e[0], e[1])):
        if k == 0: att.prompt(t, x)
        elif k == 1: att.reply(t, x)
        else:
            t0 = time.perf_counter(); att.snapshot(t, x); lat.append(time.perf_counter() - t0)
    A, cls = att.classify(final)
    return A, cls, lat, len(att.changed), len(att.label)

def offline(final, replies, turns, snaps, shared_grams, n, tau, trim=True):
    """Offline attribution (rules of demo/provenance.py; trimmed alignment as in the study code)."""
    A = tokens(final); G = ngrams(A, n); tb, tp = {}, {}
    for t, x in sorted(replies, key=lambda e: e[0]):
        for g in ngrams(tokens(x), n): tb.setdefault(g, t)
    for t, x in sorted(turns, key=lambda e: e[0]):
        for g in ngrams(tokens(x), n): tp.setdefault(g, t)
    need = set(G); te, ins = {}, {}; prev = []
    for t, text in snaps:
        if not need: break
        cur = tokens(text); m = min(len(prev), len(cur)); pre = 0
        while pre < m and prev[pre] == cur[pre]: pre += 1
        suf = 0
        while suf < m - pre and prev[-1 - suf] == cur[-1 - suf]: suf += 1
        a, b = prev[pre:len(prev) - suf], cur[pre:len(cur) - suf]
        if b:
            k = len(b) if (not a or len(b) == 1) else sum(j2 - j1 for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op in ('insert', 'replace'))
            lo, hi = max(0, pre - n + 1), min(len(cur) - n + 1, len(cur) - suf)
            for s in range(lo, max(lo, hi)):
                g = tuple(cur[s:s + n])
                if g in need: te[g] = t; ins[g] = k; need.discard(g)
        prev = cur
    lab = {}
    for g in set(G):
        bot = g in tb; pf = bot and g in tp and tp[g] < tb[g]; sh = g in shared_grams
        if bot and not pf and not sh and g in te and tb[g] <= te[g]: lab[g] = 'C'
        elif bot and (pf or (g in te and te[g] < tb[g])): lab[g] = 'E'
        elif sh: lab[g] = 'S'
        elif g in te: lab[g] = 'T' if ins[g] <= tau else 'X'
        else: lab[g] = 'X'
    rank = {c: i for i, c in enumerate(ORDER)}; best = [9] * len(A)
    for i, g in enumerate(G):
        for j in range(i, i + n): best[j] = min(best[j], rank[lab[g]])
    return A, [ORDER[b] if b < 9 else 'X' for b in best]

out = {'machine': {'python': platform.python_version(), 'processor': platform.processor() or platform.machine(), 'cpu_count': os.cpu_count(), 'note': 'one core used'}}

# ---------------- FLoRA
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
an = pd.read_pickle(f'{D}/annot.pkl'); HL = [str(s) for s in an.highlight_text.dropna()]
SHG = {g for s in HL for g in ngrams(tokens(s), 8)}
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])
OFF = {u: g['class'].tolist() for u, g in tc.groupby('user_id')}
w = pd.read_pickle(f'{D}/writing.pkl'); w = w.sort_values(['user_id', 'writing_log_time'])
WG = {u: list(zip(g.writing_log_time.astype(float) / 1e3, g.writing_content.astype(str))) for u, g in w.groupby('user_id')}; del w
users = sorted(set(d.user_id) & set(p.index) & set(OFF))
toms = lambda x: pd.Timestamp(x).value / 1e9
agree, kap, lat_all, chg, rows = [], [], [], 0, []
on_all, off_all = [], []
for u in users:
    x = d[d.user_id == u]
    ev = [(toms(t), 0, str(s)) for t, s in zip(x.user_ask_time, x.user_utterance_text)] + [(toms(t), 1, str(s)) for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text)]
    ev += [(t, 2, s) for t, s in WG.get(u, [])]
    final = str(p.loc[u, 'final_proposal'])
    t0 = time.perf_counter(); A, cls, lat, nchg, nlab = run_online(ev, HL, final, 8, 5); tsec = time.perf_counter() - t0
    t0 = time.perf_counter(); A2, cls_off = offline(final, [(e[0], e[2]) for e in ev if e[1] == 1], [(e[0], e[2]) for e in ev if e[1] == 0], WG.get(u, []), SHG, 8, 5); tsec_off = time.perf_counter() - t0
    ref = OFF[u]
    if len(ref) != len(cls): continue
    on_all += cls; off_all += ref; lat_all += lat; chg += nchg
    rows.append(dict(u=u, snaps=len(WG.get(u, [])), tokens=len(A), online_sec=tsec, offline_sec=tsec_off, U_on=cls.count('C') / len(cls), U_off=ref.count('C') / len(ref),
                     same=float(np.mean(np.array(cls) == np.array(ref))), same_recomputed=float(np.mean(np.array(cls_off) == np.array(ref))), changed=nchg, labelled=nlab))
r = pd.DataFrame(rows); on_all, off_all = np.array(on_all), np.array(off_all)
conf = {a: dict(Counter(on_all[off_all == a])) for a in ORDER}
# memory: peak traced allocation per student (separate pass, every 5th student)
mem_on, mem_off = [], []
for u in users[::5]:
    x = d[d.user_id == u]
    ev = [(toms(t), 0, str(s)) for t, s in zip(x.user_ask_time, x.user_utterance_text)] + [(toms(t), 1, str(s)) for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text)] + [(t, 2, s) for t, s in WG.get(u, [])]
    final = str(p.loc[u, 'final_proposal'])
    tracemalloc.start(); run_online(ev, HL, final, 8, 5); mem_on.append(tracemalloc.get_traced_memory()[1]); tracemalloc.stop()
    tracemalloc.start(); offline(final, [(e[0], e[2]) for e in ev if e[1] == 1], [(e[0], e[2]) for e in ev if e[1] == 0], WG.get(u, []), SHG, 8, 5); mem_off.append(tracemalloc.get_traced_memory()[1]); tracemalloc.stop()
out['flora'] = {'n_students': int(len(r)), 'snapshots': int(r.snaps.sum()),
                'token_agreement_online_vs_primary': float((on_all == off_all).mean()), 'kappa': kappa(on_all, off_all),
                'confusion_primary_to_online': conf, 'class_shares_online': {c: float((on_all == c).mean()) for c in ORDER}, 'class_shares_primary': {c: float((off_all == c).mean()) for c in ORDER},
                'Ustar_spearman_online_vs_primary': float(r[['U_on', 'U_off']].corr('spearman').iloc[0, 1]), 'Ustar_max_abs_diff': float((r.U_on - r.U_off).abs().max()),
                'Ustar_median_online': float(r.U_on.median()), 'Ustar_median_primary': float(r.U_off.median()),
                'share_labelled_ngrams_relabelled_after_entry': float(chg / r.labelled.sum()),
                'offline_recomputed_agreement_with_primary': float(np.average(r.same_recomputed, weights=r.tokens)),
                'latency_ms_per_snapshot': {'median': float(np.median(lat_all) * 1e3), 'p99': float(np.percentile(lat_all, 99) * 1e3), 'max': float(np.max(lat_all) * 1e3)},
                'online_sec_total': float(r.online_sec.sum()), 'online_sec_median_student': float(r.online_sec.median()),
                'offline_sec_total': float(r.offline_sec.sum()), 'offline_sec_median_student': float(r.offline_sec.median()),
                'peak_memory_MB_online': {'median': float(np.median(mem_on) / 2**20), 'max': float(np.max(mem_on) / 2**20), 'n': len(mem_on)},
                'peak_memory_MB_offline': {'median': float(np.median(mem_off) / 2**20), 'max': float(np.max(mem_off) / 2**20)}}
print(json.dumps(out['flora'], indent=1), flush=True)
del WG

# ---------------- RelianceScope
L = lambda f: pd.DataFrame(list(json.load(open(f'{RSD}/logs/{f}')).values())[0])
msg, ed, conv = L('messages.json'), L('edit logs.json'), L('conversations.json')
c2u = dict(zip(conv.id, conv.userId)); msg['u'] = msg.conversationId.map(c2u); ed['u'] = ed.conversationId.map(c2u)
ms = pd.read_csv(f'{RSD}/csv/measurement (by student).csv').set_index('Student ID')
rsu = sorted(set(ms.index) & set(msg.u.dropna()) & set(ed.u.dropna()))
um = msg[(msg.role == 'user') & msg.snapshot.notna()].copy(); um['code'] = um.snapshot.map(lambda s: s.get('code') if isinstance(s, dict) else None)
S = pd.concat([ed[['u', 'createdAt', 'code']], um[['u', 'createdAt', 'code']]]).dropna().sort_values(['u', 'createdAt'])
template = Counter(ed.sort_values('createdAt').groupby('u').code.first()).most_common(1)[0][0]
TG = {g for g in ngrams(tokens(template), 8)}
rows = []; lat_all = []; on_all, off_all = [], []; mem_on = []
for u in rsu:
    x = msg[msg.u == u].sort_values('createdAt'); s = S[S.u == u]
    ev = [(t / 1e3, 0 if r_ == 'user' else 1, str(c)) for t, r_, c in zip(x.createdAt, x.role, x.content)] + [(t / 1e3, 2, str(c)) for t, c in zip(s.createdAt, s.code)]
    final = ed[ed.u == u].sort_values('createdAt').code.iloc[-1]
    t0 = time.perf_counter(); A, cls, lat, nchg, nlab = run_online(ev, [template], final, 8, 20); ts = time.perf_counter() - t0
    tracemalloc.start(); run_online(ev, [template], final, 8, 20); mem_on.append(tracemalloc.get_traced_memory()[1]); tracemalloc.stop()
    A2, co = offline(final, [(e[0], e[2]) for e in ev if e[1] == 1], [(e[0], e[2]) for e in ev if e[1] == 0], [(e[0], e[2]) for e in ev if e[1] == 2], TG, 8, 20)
    on_all += cls; off_all += co; lat_all += lat
    rows.append(dict(u=u, sec=ts, U_on=cls.count('C') / len(cls), U_off=co.count('C') / len(co), changed=nchg, labelled=nlab))
r = pd.DataFrame(rows); on_all, off_all = np.array(on_all), np.array(off_all)
out['reliancescope'] = {'n_students': int(len(r)), 'token_agreement_online_vs_offline': float((on_all == off_all).mean()), 'kappa': kappa(on_all, off_all),
                        'confusion_offline_to_online': {a: dict(Counter(on_all[off_all == a])) for a in ORDER},
                        'Ustar_mean_online': float(r.U_on.mean()), 'Ustar_mean_offline': float(r.U_off.mean()), 'Ustar_spearman': float(r[['U_on', 'U_off']].corr('spearman').iloc[0, 1]),
                        'share_relabelled_after_entry': float(r.changed.sum() / r.labelled.sum()),
                        'latency_ms_per_snapshot': {'median': float(np.median(lat_all) * 1e3), 'p99': float(np.percentile(lat_all, 99) * 1e3), 'max': float(np.max(lat_all) * 1e3)},
                        'online_sec_total': float(r.sec.sum()), 'peak_memory_MB_online': {'median': float(np.median(mem_on) / 2**20), 'max': float(np.max(mem_on) / 2**20)},
                        'note': 'offline here: rules of demo/provenance.py with the template as shared source and trimmed alignment; the primary RelianceScope attribution aligns whole snapshots'}
print(json.dumps(out['reliancescope'], indent=1))
json.dump(out, open('results/r10_online.json', 'w'), indent=1, default=float)
