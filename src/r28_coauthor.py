"""Analysis log R28, A-13 to A-15 (CoAuthor), written after a review of the thirteenth version of the manuscript, after the
analyses ran. Readers, rules and parameters are those of src/r24_coauthor.py and src/r25_coauthor_range.py (this file
repeats their code; only the outputs below and the reader variant are added).
A-13 a strict lower end: chatbot-first tokens that entered the editor in an insertion of more than tau tokens (C tokens of
typing-sized insertions left out), and its coverage of the suggestion-inserted share.
A-14 conservative intervals: the 1,445 sessions come from 63 writers whom the release does not identify; each bootstrap
half-width is multiplied by sqrt(sessions / 63), the design effect if all sessions of a writer were perfectly correlated.
A-15 (variant selected_only) the replies R are only the suggestion the writer selected at each suggestion-select event (the
suggestion at currentHoverIndex of the latest suggestion-open event, which matches the inserted text in 98% of acceptances; both
texts, at the time of the suggestion-open event that showed it), instead of every suggestion shown; chatbot-first
precision and recall, the session-level rank correlation and the coverage of the range under this reader.
Usage: python -I src/r28_coauthor.py COAUTHOR_DIR [selected_only] -> results/r28_coauthor[_selected_only].json"""
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
    doc = ''; snaps = []; R = []; Q = []; prompt = ''; LAST = None
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
            LAST = (t, sg or [])
            if VAR != 'selected_only':
                for s in sg or []:
                    for k in ('original', 'trimmed'):
                        if s.get(k): R.append((t, s[k]))
        elif nm == 'suggestion-select' and VAR == 'selected_only':
            try: ix = int(e['currentHoverIndex'])   # the suggestion under the pointer when selected (currentSuggestionIndex is often 0)
            except (TypeError, ValueError): ix = -1
            if LAST is not None and 0 <= ix < len(LAST[1]):
                for k in ('original', 'trimmed'):
                    if LAST[1][ix].get(k): R.append((LAST[0], LAST[1][ix][k]))
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

rows, sess, RANGE = [], [], []
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
    after = np.zeros(len(Fu), bool)
    for i, g in enumerate(FG):
        if cls.get(g) == 'E' and g in tbot:
            te_ = sc['ted'].get(FGs[g])
            if te_ is not None and te_ >= tbot[g]: after[i:i + N] = True
    Ea = (var == 'E') & after
    Ctr = set(tt[var == 'C'].tolist()) - {-1}
    Xm = var == 'X'; xwc = Xm & np.isin(tt, list(Ctr)) if Ctr else np.zeros(len(Fu), bool)
    xsug = Xm & ~xwc & (ent == 'suggestion'); xbulk = Xm & ~xwc & (ent == 'user-insert>tau'); xoth = Xm & ~xwc & ~xsug & ~xbulk
    tsug = (var == 'T') & (ent == 'suggestion')
    m = lambda a: float(a.mean()) if len(Fu) else 0.0
    RANGE.append(dict(session=u, tokens=len(Fu), inserted=m(ent == 'suggestion'), C=m(var == 'C'), C_strict=m((var == 'C') & (te == 2)), E=m(var == 'E'), E_after=m(Ea), X=m(Xm), X_with_C=m(xwc),
                      X_suggestion_alone=m(xsug), X_writer_bulk=m(xbulk), X_other=m(xoth), T=m(var == 'T'), T_suggestion=m(tsug), S=m(var == 'S')))
    for i in range(len(Fu)): rows.append((u, i, var[i], ent[i], bool(naive[i]), int(isz[i])))
    sess.append(dict(session=u, tokens=len(Fu), U=float((var == 'C').mean()) if len(Fu) else 0.0, naive=float(naive.mean()) if len(Fu) else 0.0,
                     accepted=float(np.mean(ent == 'suggestion')) if len(Fu) else 0.0, n_shown=len(S[u]['R']), n_requests=len(S[u]['Q']),
                     n_accepted=int(sum(1 for s in sn if s[2] == 'api')), **{f'share_{c}': float((var == c).mean()) if len(Fu) else 0.0 for c in 'CESTX'}))
    G_.setdefault('INS', []).extend(ins_sizes)
if DRY: print('dry run ok', len(rows) > 0, len(sess)); sys.exit(0)


R_ = pd.DataFrame(RANGE); R_ = R_[R_.tokens > 0].reset_index(drop=True)
rng = np.random.default_rng(2501); NB = 2000
lo = R_.C; ups = {'A1-A3, N, B, R': R_.C + R_.E_after + R_.X_with_C, 'without B': R_.C + R_.E_after + R_.X,
                  'without B and A1': R_.C + R_.E + R_.X, 'without any assumption': pd.Series(np.ones(len(R_)))}
tr = R_.inserted; eps = 1e-9
def boot(v):
    v = np.asarray(v, float); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
o = {'sessions': int(len(R_)), 'mean_shares': {k: float(R_[k].mean()) for k in ['inserted', 'C', 'E', 'E_after', 'X', 'X_with_C', 'X_suggestion_alone', 'X_writer_bulk', 'X_other', 'T', 'T_suggestion', 'S']}}
cov = {}
for k, up in ups.items():
    inside = (tr >= lo - eps) & (tr <= up + eps)
    cov[k] = {'inside': boot(inside), 'below_lower_end': float((tr < lo - eps).mean()), 'above_upper_end': float((tr > up + eps).mean()),
              'mean_upper_end': float(up.mean()), 'mean_shortfall_when_above': float((tr - up)[tr > up + eps].mean()) if (tr > up + eps).any() else 0.0,
              'cohort_at_or_above_half': {'lower': float((lo >= .5).mean()), 'upper': float((up >= .5).mean())}}
o['A11_coverage'] = cov
o['A11_cohort_truth_at_or_above_half'] = float((tr >= .5).mean())
o['A11_note'] = 'sessions are resampled; the 1,445 sessions come from 63 writers whom the release does not identify, so the intervals are too narrow'
X_ = R_.X.mean()
o['A12_X_subtypes_share_of_X'] = {k: float(R_[k].mean() / X_) for k in ['X_with_C', 'X_suggestion_alone', 'X_writer_bulk', 'X_other']}
o['A12_suggestion_text_in_T_mean_share_of_document'] = float(R_.T_suggestion.mean())
o['A12_sessions_above_upper_end_explained'] = {'with_any_X_suggestion_alone_or_T_suggestion': float(((tr > ups['A1-A3, N, B, R'] + eps) & ((R_.X_suggestion_alone > 0) | (R_.T_suggestion > 0))).sum() / max(((tr > ups['A1-A3, N, B, R'] + eps)).sum(), 1))}

# A-13 strict lower end
lo_s = R_.C_strict
o['A13_strict_lower_end'] = {'mean_C': float(R_.C.mean()), 'mean_C_strict': float(lo_s.mean()),
    'below_lower_end': {'C': float((tr < R_.C - eps).mean()), 'C_strict': float((tr < lo_s - eps).mean())},
    'inside': {k: boot((tr >= lo_s - eps) & (tr <= up + eps)) for k, up in ups.items()},
    'cohort_at_or_above_half_lower': {'C': float((R_.C >= .5).mean()), 'C_strict': float((lo_s >= .5).mean())}}
# A-14 conservative intervals (writers not identified)
fac = float(np.sqrt(len(R_) / 63))
def cons(b): return [b[0], max(0.0, b[0] - fac * (b[0] - b[1])), min(1.0, b[0] + fac * (b[2] - b[0]))]
o['A14_conservative'] = {'factor': fac, 'inside': {k: cons(v['inside']) for k, v in cov.items()}}
# A-15 (also computed for the main reader): precision, recall, session rank correlation
T_ = pd.DataFrame(rows, columns=['session', 'i', 'cls', 'entry', 'naive', 'ins_size']); SD = pd.DataFrame(sess); SD = SD[SD.tokens > 0]
acc = T_.entry == 'suggestion'
from scipy import stats
o['A15_reader'] = {'variant': VAR or 'all_shown', 'C_precision': float(acc[T_.cls == 'C'].mean()), 'naive_precision': float(acc[T_.naive].mean()),
    'recall_C_of_suggestion_tokens': float((T_.cls[acc] == 'C').mean()), 'recall_naive_of_suggestion_tokens': float(T_.naive[acc].mean()),
    'recall_C_suggestions_ge_8': float((T_.cls[acc & (T_.ins_size >= 8)] == 'C').mean()),
    'spearman_U_accepted': float(stats.spearmanr(SD.U, SD.accepted).correlation), 'spearman_naive_accepted': float(stats.spearmanr(SD.naive, SD.accepted).correlation),
    'mean_U': float(SD.U.mean()), 'mean_accepted': float(SD.accepted.mean()), 'class_shares_mean': {c: float(SD[f'share_{c}'].mean()) for c in 'CESTX'}}
o['per_session'] = R_.to_dict(orient='records')
json.dump(o, open(os.path.join(HERE, '..', 'results', 'r28_coauthor' + ('_' + VAR if VAR else '') + '.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in o.items() if k != 'per_session'}, indent=1, default=float))
