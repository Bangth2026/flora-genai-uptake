"""Analysis log R24, A-1 to A-9: the frozen attribution applied to CoAuthor (Lee, Liang & Yang, 2022; 1,445 writing
sessions of 63 writers with GPT-3 suggestions; file coauthor-v1.0.zip from https://coauthor.stanford.edu), whose
platform logged every accepted suggestion as an event of its own. This replaces NIRVANA as the hold-out set.

The attribution is not re-implemented: the functions scan, classify, project and propagate are read from
src/r9b_flora_token_entry.py and executed as they are (n = 8, tau = 5, delta = 0, peer threshold 5, token priority
C > E > S > T > X, token entry rule). Only the readers are new:
  snapshots  the document after every text-insert or text-delete event, rebuilt from the system-initialize document and
             the Quill deltas of the events
  replies R  every suggestion shown at a suggestion-open event (fields 'original' and 'trimmed'), at that event's time
  turns Q    the document text before the cursor at each suggestion-get event (the context sent to the model)
  shared H   8-grams of the session's initial prompt (the system-initialize document) plus the peer rule over the final
             documents
  document A the final document with the initial prompt removed when the final document still begins with it
Times are milliseconds on the platform's clock, converted to seconds.

Independent signal: a token of A is suggestion-inserted if the snapshot transition that last inserted it is a
text-insert event with eventSource 'api', which the platform writes when the writer accepts a suggestion. The signal
does not read any text match. It misses suggestion text that the writer retyped.
Entry routes of a token (by the transition that last inserted it): 'suggestion' (api text-insert), 'typed' (user
text-insert of at most tau tokens), 'user-insert>tau' (user text-insert of more than tau tokens: pastes and other
bulk insertions), 'prompt' (the initial document), 'other'.
Usage: python -I src/r24_coauthor.py COAUTHOR_DIR [variant] -> results/r24_coauthor[_variant].json
variants: (none) main run; noprompt_H (initial prompt not in H); trimmed_only (only 'trimmed' suggestion texts as R);
tauK (tau = K, for the data-driven threshold of A-9); posthoc_len (main parameters, recall by size of the accepted
suggestion, written after the pre-specified results were seen)"""
import sys, os, re, ast, json, glob
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
HERE = os.path.dirname(os.path.abspath(__file__)); DIRX = sys.argv[1]; VAR = sys.argv[2] if len(sys.argv) > 2 else ''
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
src = open(os.path.join(HERE, 'r9b_flora_token_entry.py')).read()
def block(name):
    i = src.index(f'def {name}('); lines = src[i:].split('\n'); out = [lines[0]]
    for ln in lines[1:]:
        if ln and not ln[0].isspace(): break
        out.append(ln)
    return '\n'.join(out)
G_ = {'np': np, 'pd': pd, 'SequenceMatcher': SequenceMatcher, 'Counter': Counter, 'grams': grams}
for nm in ('scan', 'classify', 'project', 'propagate'): exec(block(nm), G_)
scan, classify, project, propagate = (G_[k] for k in ('scan', 'classify', 'project', 'propagate'))
N, TAU = 8, (int(VAR[3:]) if VAR.startswith('tau') else 5)
files = sorted(glob.glob(os.path.join(DIRX, '*.jsonl')))
DRY = os.environ.get('DRY') == '1'
if DRY: files = files[:3]

def apply_delta(doc, delta):
    out, pos = [], 0
    for op in delta.get('ops', []):
        if 'retain' in op: out.append(doc[pos:pos + op['retain']]); pos += op['retain']
        elif 'delete' in op: pos += op['delete']
        elif 'insert' in op:
            ins = op['insert']; out.append(ins if isinstance(ins, str) else '')
    out.append(doc[pos:]); return ''.join(out)

def read(f):
    ev = [json.loads(l) for l in open(f)]
    doc = ''; snaps = []; R = []; Q = []; prompt = ''
    for e in ev:
        t = float(e['eventTimestamp']) / 1000.0; nm = e['eventName']
        if nm == 'system-initialize':
            doc = e['currentDoc'] or ''; prompt = doc; snaps.append((t, doc, 'prompt', 0)); continue
        if nm in ('text-insert', 'text-delete'):
            d = e['textDelta']; d = ast.literal_eval(d) if isinstance(d, str) and d else (d or {})
            new = apply_delta(doc, d)
            ins = ''.join(op['insert'] for op in d.get('ops', []) if isinstance(op.get('insert'), str))
            kind = 'api' if (nm == 'text-insert' and e['eventSource'] == 'api') else ('user-insert' if nm == 'text-insert' else 'delete')
            doc = new; snaps.append((t, doc, kind, len(tok(ins))))
        elif nm == 'suggestion-get':
            cur = int(e['currentCursor']) if str(e['currentCursor']).isdigit() else len(doc); Q.append((t, doc[:cur]))
        elif nm == 'suggestion-open':
            sg = e['currentSuggestions']; sg = ast.literal_eval(sg) if isinstance(sg, str) else sg
            for s in sg or []:
                for k in (('trimmed',) if VAR == 'trimmed_only' else ('original', 'trimmed')):
                    if s.get(k): R.append((t, s[k]))
    return dict(snaps=snaps, R=R, Q=Q, prompt=prompt)

S = {}
for f in files:
    S[os.path.basename(f)[:-6]] = read(f)
users = sorted(S)
def final_tokens(u):
    fin = tok(S[u]['snaps'][-1][1]) if S[u]['snaps'] else []; pr = tok(S[u]['prompt'])
    return (fin[len(pr):], len(pr)) if pr and fin[:len(pr)] == pr else (fin, 0)
F = {u: final_tokens(u) for u in users}
H8 = set() if VAR == 'noprompt_H' else {g for u in users for g in grams(tok(S[u]['prompt']), N)}
Cc8 = Counter(g for u in users for g in set(grams(F[u][0], N)))
G_['H'] = {N: H8}; G_['Cc'] = {N: Cc8}

rows, sess = [], []
for u in users:
    sn = S[u]['snaps']
    if not sn: continue
    times = [s[0] for s in sn]; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(s[1])], np.int32) for s in sn]
    Fu, off = F[u]; full = tok(sn[-1][1])
    Fi = np.array([vocab.setdefault(x, len(vocab)) for x in Fu], np.int32)
    tbot, tpr = {}, {}
    for t, s in sorted(S[u]['R']):
        for g in grams(tok(s), N): tbot.setdefault(g, t)
    for t, s in sorted(S[u]['Q']):
        for g in grams(tok(s), N): tpr.setdefault(g, t)
    FG = grams(Fu, N); FGs = {g: tuple(Fi[i:i + N].tolist()) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), N, set(FGs.values()))
    cls = classify(FGs, tbot, tpr, sc, N, delta=0); lab = project(Fu, N, cls)
    last, tl, ti = propagate(enc, TAU)
    te = np.full(len(Fu), 2, np.int8); tt = np.full(len(Fu), -1, np.int64)
    Fall = np.array([vocab.setdefault(x, len(vocab)) for x in full], np.int32)
    te_all = np.full(len(full), 2, np.int8); tt_all = np.full(len(full), -1, np.int64)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, Fall.tolist(), last.tolist(), autojunk=False).get_opcodes():
        if op == 'equal': te_all[i1:i2] = tl[j1:j2]; tt_all[i1:i2] = ti[j1:j2]
    te, tt = te_all[off:off + len(Fu)], tt_all[off:off + len(Fu)]
    var = np.where(np.isin(lab, ['C', 'E', 'S']), lab, np.where(te == 1, 'T', 'X'))
    def entry(k):
        if k < 0: return 'other'
        kind, n_ins = sn[k][2], sn[k][3]
        if kind == 'api': return 'suggestion'
        if kind == 'prompt': return 'prompt'
        if kind == 'user-insert': return 'typed' if n_ins <= TAU else 'user-insert>tau'
        return 'other'
    ent = np.array([entry(int(k)) for k in tt], dtype=object)
    isz = np.array([sn[int(k)][3] if k >= 0 else -1 for k in tt])
    naive = np.zeros(len(Fu), bool)
    for i, g in enumerate(FG):
        if g in tbot: naive[i:i + N] = True
    ins_sizes = [s[3] for s in sn if s[2] == 'user-insert' and s[3] > 0]
    for i in range(len(Fu)): rows.append((u, i, var[i], ent[i], bool(naive[i]), int(isz[i])))
    sess.append(dict(session=u, tokens=len(Fu), U=float((var == 'C').mean()) if len(Fu) else 0.0, naive=float(naive.mean()) if len(Fu) else 0.0,
                     accepted=float(np.mean(ent == 'suggestion')) if len(Fu) else 0.0, n_shown=len(S[u]['R']), n_requests=len(S[u]['Q']),
                     n_accepted=int(sum(1 for s in sn if s[2] == 'api')), **{f'share_{c}': float((var == c).mean()) if len(Fu) else 0.0 for c in 'CESTX'}))
    G_.setdefault('INS', []).extend(ins_sizes)
if DRY: print('dry run ok', len(rows) > 0, len(sess)); sys.exit(0)

from scipy import stats
T_ = pd.DataFrame(rows, columns=['session', 'i', 'cls', 'entry', 'naive', 'ins_size'])
SD = pd.DataFrame(sess); SD = SD[SD.tokens > 0].reset_index(drop=True)
rng = np.random.default_rng(2401); NB = 2000
o = {'variant': VAR or 'main', 'sessions': int(len(SD)), 'tokens': int(len(T_)), 'parameters': {'n': N, 'tau': TAU, 'delta': 0, 'peer_threshold': 5},
     'shared_8grams_from_prompts': len(H8), 'class_shares_mean': {c: float(SD[f'share_{c}'].mean()) for c in 'CESTX'},
     'entry_shares': T_.entry.value_counts(normalize=True).to_dict()}
ct = pd.crosstab(T_.cls, T_.entry); o['class_by_entry_counts'] = {c: {k: int(v) for k, v in ct.loc[c].items()} for c in ct.index}
acc = T_.entry == 'suggestion'
o['A1_C_tokens_suggestion_inserted'] = float(acc[T_.cls == 'C'].mean()); o['A1_C_tokens_typed'] = float((T_.entry[T_.cls == 'C'] == 'typed').mean())
o['A2_suggestion_tokens_by_class'] = T_[acc].cls.value_counts(normalize=True).to_dict()
o['A3_share_suggestion_inserted'] = {c: float(acc[T_.cls == c].mean()) for c in 'ESTX'}
def boot_rho(a, b):
    v = []
    for _ in range(NB):
        i = rng.integers(0, len(SD), len(SD)); v.append(stats.spearmanr(SD[a].values[i], SD[b].values[i]).correlation)
    return [float(stats.spearmanr(SD[a], SD[b]).correlation), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
d_ = []
for _ in range(NB):
    i = rng.integers(0, len(SD), len(SD)); x = SD.iloc[i]
    d_.append(stats.spearmanr(x.U, x.accepted).correlation - stats.spearmanr(x.naive, x.accepted).correlation)
o['A4_session'] = {'spearman_U_accepted': boot_rho('U', 'accepted'), 'spearman_naive_accepted': boot_rho('naive', 'accepted'),
                   'U_minus_naive': [float(np.nanmean(d_)), float(np.nanpercentile(d_, 2.5)), float(np.nanpercentile(d_, 97.5))],
                   'mae_U_accepted': float((SD.U - SD.accepted).abs().mean()), 'mae_naive_accepted': float((SD.naive - SD.accepted).abs().mean()),
                   'mean_U': float(SD.U.mean()), 'mean_accepted': float(SD.accepted.mean()), 'mean_naive': float(SD.naive.mean()),
                   'share_ge_half': {k: float((SD[k] >= .5).mean()) for k in ('U', 'accepted', 'naive')},
                   'agreement_at_half': {k: float(((SD[k] >= .5) == (SD.accepted >= .5)).mean()) for k in ('U', 'naive')},
                   'sessions_with_any_accepted': int((SD.n_accepted > 0).sum())}
o['A5_naive_on_signal'] = {'accepted_tokens_marked': float(T_.naive[acc].mean()), 'marked_tokens_accepted': float(acc[T_.naive].mean()) if T_.naive.any() else None,
                           'C_tokens_accepted': float(acc[T_.cls == 'C'].mean())}
o['A6_strata'] = {'suggestion_stratum': {'tokens': int(acc.sum()), 'by_class': T_[acc].cls.value_counts().to_dict()},
                  'other_tokens': {'tokens': int((~acc).sum()), 'by_class': T_[~acc].cls.value_counts().to_dict(),
                                   'C_by_entry': T_[(~acc) & (T_.cls == 'C')].entry.value_counts().to_dict()}}
E_ = T_.cls == 'E'
o['A7_echo'] = {'E_tokens': int(E_.sum()), 'E_suggestion_inserted': int((E_ & acc).sum()), 'E_naive_marked': int((E_ & T_.naive).sum())}
# A8 coincidence: C without a suggestion event; typed C runs
runs = []
for u, g in T_.groupby('session', sort=False):
    c = ((g.cls == 'C') & (g.entry == 'typed')).values; i = 0
    while i < len(c):
        if c[i]:
            j = i
            while j < len(c) and c[j]: j += 1
            runs.append(j - i); i = j
        else: i += 1
runs = np.array(runs) if runs else np.zeros(0)
nC = int((T_.cls == 'C').sum())
o['A8_coincidence'] = {'C_tokens': nC, 'C_without_suggestion_event': int(((T_.cls == 'C') & ~acc).sum()), 'bound_share_of_C': float(((T_.cls == 'C') & ~acc).sum() / max(nC, 1)),
                       'typed_C_tokens': int(runs.sum()), 'typed_C_runs': int(len(runs)), 'typed_C_median_run': float(np.median(runs)) if len(runs) else None,
                       'typed_C_tokens_in_runs_lt_12': int(runs[runs < 12].sum()), 'typed_C_tokens_in_runs_ge_16': int(runs[runs >= 16].sum())}
INS = np.array(G_.get('INS', []))
w = np.sort(INS); cw = np.cumsum(w) / w.sum() if len(w) else np.zeros(0)
o['A9_insert_sizes'] = {'user_inserts': int(len(INS)), 'share_gt_tau': float((INS > TAU).mean()) if len(INS) else None,
                        'p99_by_insert': float(np.percentile(INS, 99)) if len(INS) else None,
                        'p99_by_token': float(w[np.searchsorted(cw, .99)]) if len(w) else None,
                        'X_entry': T_[T_.cls == 'X'].entry.value_counts(normalize=True).to_dict()}
if VAR == 'posthoc_len':   # post hoc, written after the pre-specified results were seen: recall by length of the accepted suggestion
    a_ = T_[acc]; bins = [(1, 5), (6, 7), (8, 15), (16, 10 ** 6)]
    o['A10_posthoc_recall_by_insert_size'] = {f'{lo}-{hi}' if hi < 10 ** 6 else f'>={lo}': {'tokens': int(((a_.ins_size >= lo) & (a_.ins_size <= hi)).sum()),
        **{c: float((a_[(a_.ins_size >= lo) & (a_.ins_size <= hi)].cls == c).mean()) for c in 'CETXS'}} for lo, hi in bins}
o['per_session'] = SD.to_dict(orient='records')
name = 'r24_coauthor' + ('_' + VAR if VAR else '') + '.json'
json.dump(o, open(os.path.join(HERE, '..', 'results', name), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in o.items() if k != 'per_session'}, indent=1, default=float))
