"""Analysis log R13, F-12 and F-14 (FLoRA).

F-12. Selection in the chatbot panel before entry for the passages audited under the token entry rule (log entry R11,
      F-5: the sample drawn by src/r11_flora.py from runs of at least eight T or X tokens, random_state 2091), with the
      keystroke criterion recomputed as a check that the sample is the same.
F-14. Unresolved text that entered with the first editor snapshot: size of the first snapshot, whether the first
      snapshots of different students share text (a default text loaded into the editor would appear in all of
      them), whether the first snapshot came before the student's first chatbot reply, and how often its 8-grams occur
      in other students' proposals below the peer threshold of the shared-source rule (1 to 4 other proposals).
Classes are read from results/flora_token_classes.csv.gz (token entry rule).
Usage: python -I src/r13_flora.py DATA_DIR -> results/r13_flora.json
DATA_DIR needs dialogue.xlsx, proposal.xlsx, writing.pkl and trace.pkl (Figshare files 5.1-5.4, 5.2 as one table)."""
import sys, re, json
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
D = sys.argv[1]; NB = 2000; rng = np.random.default_rng(2160)
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
users = sorted(set(d.user_id) & set(p.index))
P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])
CL = {u: g['class'].values for u, g in tc.groupby('user_id')}
assert all(len(CL[u]) == len(P[u]) for u in users)
REP0 = {u: pd.to_datetime(x.chatbot_utterance_time).min() for u, x in d.groupby('user_id')}


def propagate(enc, tau=5):
    prev = np.zeros(0, np.int32); lab = np.zeros(0, np.int8); tin = np.zeros(0, np.int64)
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
            ops = [('insert', 0, 0, 0, len(b))] if len(a) == 0 else SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes()
            for op, i1, i2, j1, j2 in ops:
                if op == 'equal': nl[pre + j1:pre + j2] = lab[pre + i1:pre + i2]; nt[pre + j1:pre + j2] = tin[pre + i1:pre + i2]
                elif op in ('insert', 'replace'): nl[pre + j1:pre + j2] = 1 if (j2 - j1) <= tau else 2; nt[pre + j1:pre + j2] = k
        lab, tin, prev = nl, nt, cur
    return prev, lab, tin


w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
TK = {}; FIRST = {}
for u, g_ in w[w.user_id.isin(users)].groupby('user_id'):
    times = list(g_.t); vocab = {}; contents = list(g_.writing_content)
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in contents]
    F = P[u]; Fi = np.array([vocab.setdefault(x, len(vocab)) for x in F], np.int32)
    last, tl, ti = propagate(enc)
    tt = np.full(len(F), -1, np.int64)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, Fi.tolist(), last.tolist(), autojunk=False).get_opcodes():
        if op == 'equal': tt[i1:i2] = ti[j1:j2]
    TK[u] = dict(tt=tt, times=times); FIRST[u] = (times[0], tok(contents[0]))
del w
o = {'n_students': len(users)}

# ---------- F-14 unresolved text entering with the first snapshot
fs_students = Counter()   # 8-gram -> number of students whose first snapshot contains it
for u, (_, z) in FIRST.items():
    for g in set(grams(z, 8)): fs_students[g] += 1
Cc = Counter(g for u in users for g in set(grams(P[u], 8)))
lens = np.array([len(z) for _, z in FIRST.values()])
X0 = {u: (CL[u] == 'X') & (TK[u]['tt'] == 0) for u in TK}
nX = sum(int((CL[u] == 'X').sum()) for u in users); nX0 = sum(int(m.sum()) for m in X0.values())
st0 = [u for u in TK if X0[u].any()]
# 8-grams wholly inside first-snapshot X text, by the number of OTHER proposals that contain them
rec = Counter(); shared_first = Counter(); tot_g = 0
tok_rec = Counter(); tok_fs = 0
for u in st0:
    F = P[u]; m = X0[u]; cov_other = np.zeros(len(F), bool); cov_fs = np.zeros(len(F), bool)
    for i, g in enumerate(grams(F, 8)):
        if m[i:i + 8].all():
            k = Cc[g] - 1; tot_g += 1; rec['0' if k == 0 else ('1-4' if k <= 4 else '>=5')] += 1
            if k >= 1: cov_other[i:i + 8] = True
            if fs_students[g] - 1 >= 1: shared_first['in another student first snapshot'] += 1; cov_fs[i:i + 8] = True
    tok_rec['X0_tokens_in_8gram_of_another_proposal'] += int((cov_other & m).sum()); tok_fs += int((cov_fs & m).sum())
before_reply = [u for u in st0 if u in REP0 and FIRST[u][0] < REP0[u]]
x0_before = sum(int(X0[u].sum()) for u in before_reply)
# first snapshots that share any 8-gram with the first snapshots of at least 5 other students (a default text)
tmpl = [u for u, (_, z) in FIRST.items() if any(fs_students[g] - 1 >= 5 for g in set(grams(z, 8)))]
o['F14_first_snapshot'] = {
    'share_of_X_tokens_entering_with_first_snapshot': nX0 / nX,
    'students_with_X_in_first_snapshot': len(st0),
    'first_snapshot_tokens_quartiles_all_students': np.percentile(lens, [25, 50, 75]).tolist(),
    'share_of_students_first_snapshot_empty': float(np.mean(lens == 0)),
    'share_of_students_first_snapshot_gt_100_tokens': float(np.mean(lens > 100)),
    'first_snapshot_tokens_quartiles_students_with_X0': np.percentile([len(FIRST[u][1]) for u in st0], [25, 50, 75]).tolist(),
    'students_whose_first_snapshot_shares_an_8gram_with_ge5_other_first_snapshots': len(tmpl),
    'X0_8grams_by_number_of_other_proposals_containing_them': dict(rec), 'X0_8grams_total': tot_g,
    'X0_8grams_in_another_student_first_snapshot': shared_first['in another student first snapshot'],
    'X0_tokens_covered_by_8gram_in_1_to_4_other_proposals_share': tok_rec['X0_tokens_in_8gram_of_another_proposal'] / nX0 if nX0 else None,
    'X0_tokens_covered_by_8gram_in_another_first_snapshot_share': tok_fs / nX0 if nX0 else None,
    'students_with_X0_whose_first_snapshot_precedes_first_reply': len(before_reply),
    'share_of_X0_tokens_from_those_students': x0_before / nX0 if nX0 else None}
print('F14', json.dumps(o['F14_first_snapshot'], indent=1), flush=True)

# ---------- F-12 selection in the chatbot panel, token entry rule audit sample (R11 F-5)
tr = pd.read_pickle(f'{D}/trace.pkl'); tr['t'] = pd.to_datetime(tr.log_trace_time, unit='ms')
ek = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')][['user_id', 't', 'event_value']].copy()
sel = tr[(tr.learning_action_label == 'READ_FEEDBACK_MULTI_AGENTS') & tr.event_value.astype(str).str.startswith('MOUSE_SELECT:::')][['user_id', 't', 'event_value']].copy()
del tr
v = ek.event_value.astype(str).str.replace('KEY:::', '', regex=False); key = v.str.split('---').str[0]
ek['printable'] = key.str.len().eq(1).values
EKT = {u: np.sort(x.t.values[x.printable.values]) for u, x in ek.groupby('user_id')}
KEYU = sorted(set(EKT) & set(users))
SEL = {u: [(t_, set(grams(tok(s.replace('MOUSE_SELECT:::', '')), 8))) for t_, s in zip(x.t, x.event_value.astype(str))] for u, x in sel.groupby('user_id')}
runs = []
for u in KEYU:
    L = CL[u]; i = 0
    while i < len(L):
        j = i
        while j < len(L) and L[j] == L[i]: j += 1
        if j - i >= 8 and L[i] in 'TX': runs.append((u, L[i], i, j))
        i = j
runs = pd.DataFrame(runs, columns=['u', 'cls', 'i', 'j']); o['F12_selection_token_entry_sample'] = {}
for c in 'TX':
    x = runs[runs.cls == c].sample(frac=1, random_state=2091).groupby('u').head(3).head(50); ag, sl = [], []
    for r in x.itertuples():
        tt = TK[r.u]['tt']; times = TK[r.u]['times']; ks = tt[r.i:r.j]; ks = ks[ks >= 0]
        pg = set(grams(P[r.u][r.i:r.j], 8))
        if not len(ks):
            ag.append(c == 'X'); sl.append(any(sg & pg for _, sg in SEL.get(r.u, []))); continue
        t1 = np.datetime64(times[ks.min()]) - np.timedelta64(3, 's'); t2 = np.datetime64(times[ks.max()]) + np.timedelta64(1, 's')
        if ks.min() > 0: t1 = min(t1, np.datetime64(times[ks.min() - 1]))
        ekt = EKT.get(r.u, np.array([], 'datetime64[ns]')); keys = int(((ekt >= t1) & (ekt <= t2)).sum())
        chars = len(' '.join(P[r.u][r.i:r.j]))
        ag.append(keys >= 0.5 * chars if c == 'T' else keys < 0.2 * chars)
        sl.append(any(np.datetime64(ts) <= t2 and (sg & pg) for ts, sg in SEL.get(r.u, [])))
    x = x.assign(agree=np.array(ag, float), selected=np.array(sl, float)); us_ = x.u.unique(); bs = []
    for _ in range(NB):
        b = rng.choice(us_, len(us_)); bs.append(pd.concat([x[x.u == q_] for q_ in b]).selected.mean())
    o['F12_selection_token_entry_sample'][c] = {'passages': int(len(x)), 'students': int(len(us_)),
        'keystroke_criterion_agree_check': float(x.agree.mean()), 'selected_in_chat_before_entry': float(x.selected.mean()),
        'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
print('F12', o['F12_selection_token_entry_sample'], flush=True)
json.dump(o, open('results/r13_flora.json', 'w'), indent=1, default=float)
