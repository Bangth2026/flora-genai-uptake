"""Analysis log R22, N-1 to N-5: the frozen attribution applied, unchanged, to NIRVANA (Jelson et al., 2026), a second
public dataset with an independent copy signal (77 students, an analytical essay, GPT-3.5-turbo; OSF project 3a8uh,
file dataset.xlsx, md5 cf7d122ddf299b6885f83c200a3ef035).

The attribution is not re-implemented: the functions scan, classify, project and propagate are read from
src/r9b_flora_token_entry.py and executed as they are (n = 8, tau = 5, delta = 0, peer threshold 5, token priority C > E
> S > T > X, token entry rule). Only the readers are new:
  replies    rows with op_type gpt_response (time, selected_text)
  prompts    rows with op_type gpt_inquiry (time, selected_text)
  snapshots  current_editor after every editor operation that has one, lines joined by a line break
  final text the last snapshot
  shared H   8-grams of every text copied from the essay-prompt pane (op_loc essay_prompt), plus 8-grams that occur in
             the final texts of at least five other students (the peer rule)
Times are seconds on the platform's clock; replies and prompts are on the same clock as the editor.

Copy signal, independent of the rules: every editor paste is matched to the latest earlier copy or cut event (any pane)
whose tokens cover at least half of the tokens the paste inserted (difflib matching blocks); its source is that event's
pane (gpt = a ChatGPT reply, editor, essay_prompt) or 'unmatched'. A token of the final text is ChatGPT-copied if the
snapshot transition that last inserted it (propagate) is a paste whose source is gpt.
Usage: python -I src/r22_nirvana.py NIRVANA_XLSX -> results/r22_nirvana.json, results/r22_nirvana_tokens.csv.gz"""
import sys, os, re, ast, json
import numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
HERE = os.path.dirname(os.path.abspath(__file__)); X = sys.argv[1]
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
# ---- the frozen functions, executed from the study code
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
N, TAU = 8, 5
# ---- readers
e = pd.read_excel(X, sheet_name='Essay Data').sort_values(['essay_num', 'op_index'])
part = pd.read_excel(X, sheet_name='Participants').set_index('id')
users = sorted(e.essay_num.unique())
DRY = os.environ.get('DRY') == '1'
if DRY: users = users[:3]
def snap_text(v):
    try: return '\n'.join(ast.literal_eval(v))
    except Exception: return None
S = {}
for u, g in e.groupby('essay_num'):
    ed = g[(g.op_loc == 'editor') & g.current_editor.notna()]
    rows = [(float(t), snap_text(v), int(oi), ot) for t, v, oi, ot in zip(ed.time, ed.current_editor, ed.op_index, ed.op_type)]
    S[u] = [r for r in rows if r[1] is not None]
F = {u: tok(S[u][-1][1]) if S[u] else [] for u in users}
H8 = {g for s in e[(e.op_loc == 'essay_prompt') & e.op_type.isin(['y', 'x'])].selected_text.dropna() for g in grams(tok(s), N)}
Cc8 = Counter(g for u in users for g in set(grams(F[u], N)))
G_['H'] = {N: H8}; G_['Cc'] = {N: Cc8}
def dialogue(u):
    x = e[e.essay_num == u]; tbot, tpr = {}, {}
    for t, s in zip(x[x.op_type == 'gpt_response'].time, x[x.op_type == 'gpt_response'].selected_text):
        for g_ in grams(tok(s), N): tbot.setdefault(g_, float(t))
    for t, s in zip(x[x.op_type == 'gpt_inquiry'].time, x[x.op_type == 'gpt_inquiry'].selected_text):
        for g_ in grams(tok(s), N): tpr.setdefault(g_, float(t))
    return tbot, tpr
# ---- paste sources (copy signal)
def paste_sources(u):
    g = e[e.essay_num == u]; copies = []; out = {}
    prev_tokens = []
    snap_by_op = {oi: txt for (t, txt, oi, ot) in S[u]}
    order = list(g.op_index)
    last_snap = []
    for _, r in g.iterrows():
        if r.op_type in ('y', 'x') and pd.notna(r.selected_text): copies.append((r.op_loc, tok(r.selected_text)))
        if r.op_loc == 'editor' and int(r.op_index) in snap_by_op:
            cur = tok(snap_by_op[int(r.op_index)])
            if r.op_type == 'p':
                ins = [cur[j] for op, i1, i2, j1, j2 in SequenceMatcher(None, last_snap, cur, autojunk=False).get_opcodes() if op in ('insert', 'replace') for j in range(j1, j2)]
                srcp = 'unmatched'
                for loc, ct in reversed(copies):
                    if not ins or not ct: continue
                    m = sum(sz for a, b, sz in SequenceMatcher(None, ct, ins, autojunk=False).get_matching_blocks())
                    if m >= 0.5 * len(ins): srcp = loc; break
                out[int(r.op_index)] = (srcp, len(ins))
            last_snap = cur
    return out
rows, stud = [], []
for u in users:
    if not S[u]: continue
    times = [s[0] for s in S[u]]; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(s[1])], np.int32) for s in S[u]]
    Fu = F[u]; Fi = np.array([vocab.setdefault(x, len(vocab)) for x in Fu], np.int32)
    tbot, tpr = dialogue(u); FG = grams(Fu, N); FGs = {g_: tuple(Fi[i:i + N].tolist()) for i, g_ in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), N, set(FGs.values()))
    cls = classify(FGs, tbot, tpr, sc, N, delta=0); lab = project(Fu, N, cls)
    last, tl, ti = propagate(enc, TAU)
    te = np.full(len(Fu), 2, np.int8); tt = np.full(len(Fu), -1, np.int64)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, Fi.tolist(), last.tolist(), autojunk=False).get_opcodes():
        if op == 'equal': te[i1:i2] = tl[j1:j2]; tt[i1:i2] = ti[j1:j2]
    var = np.where(np.isin(lab, ['C', 'E', 'S']), lab, np.where(te == 1, 'T', 'X'))
    ps = paste_sources(u)
    def entry(k):
        if k < 0: return 'unknown'
        oi, ot = S[u][k][2], S[u][k][3]
        if ot == 'p': return 'paste:' + ps.get(oi, ('unmatched', 0))[0]
        if ot in ('i', 'd_i'): return 'typed'
        return 'other:' + str(ot)
    ent = [entry(int(k)) for k in tt]
    naive = np.zeros(len(Fu), bool)
    for i, g_ in enumerate(FG):
        if g_ in tbot: naive[i:i + N] = True
    for i in range(len(Fu)): rows.append((u, i, Fu[i], var[i], ent[i], bool(naive[i])))
    stud.append(dict(user=int(u), tokens=len(Fu), U=float((var == 'C').mean()), naive=float(naive.mean()),
                     copied=float(np.mean([x == 'paste:gpt' for x in ent])) if len(ent) else 0.0,
                     gpt_final_pct=float(part.loc[u, 'GPT Final %']), n_replies=int((e[e.essay_num == u].op_type == 'gpt_response').sum()),
                     **{f'share_{c}': float((var == c).mean()) for c in 'CESTX'}))
if DRY: print('dry run ok', len(rows) > 0); sys.exit(0)
T_ = pd.DataFrame(rows, columns=['user', 'token_index', 'token', 'cls', 'entry', 'naive_cover'])
SD = pd.DataFrame(stud)
os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
T_.to_csv(os.path.join(HERE, '..', 'results', 'r22_nirvana_tokens.csv.gz'), index=False)
o = {'n_students': int(len(SD)), 'n_tokens': int(len(T_)), 'parameters': {'n': N, 'tau': TAU, 'delta': 0, 'peer_threshold': 5},
     'n_shared_8grams_prompt_pane': len(H8)}
o['class_shares_mean'] = {c: float(SD[f'share_{c}'].mean()) for c in 'CESTX'}
o['entry_shares'] = T_.entry.value_counts(normalize=True).to_dict()
ct = pd.crosstab(T_.cls, T_.entry); o['class_by_entry_counts'] = {c: ct.loc[c].to_dict() for c in ct.index}
cp = T_.entry == 'paste:gpt'
o['N1_C_tokens_entered_by_chatgpt_copy'] = float(cp[T_.cls == 'C'].mean())
o['N1_C_tokens_typed'] = float((T_.entry[T_.cls == 'C'] == 'typed').mean())
o['N2_chatgpt_copied_tokens_by_class'] = T_[cp].cls.value_counts(normalize=True).to_dict()
o['N3_E_tokens_chatgpt_copied'] = float(cp[T_.cls == 'E'].mean()); o['N3_X_tokens_chatgpt_copied'] = float(cp[T_.cls == 'X'].mean())
o['N3_T_tokens_chatgpt_copied'] = float(cp[T_.cls == 'T'].mean())
from scipy import stats
o['N4_student'] = {'spearman_U_vs_gpt_final_pct': float(stats.spearmanr(SD.U, SD.gpt_final_pct).correlation),
                   'spearman_U_vs_copied_share': float(stats.spearmanr(SD.U, SD.copied).correlation),
                   'mae_U_vs_gpt_final_pct': float((SD.U - SD.gpt_final_pct).abs().mean()),
                   'mean_U': float(SD.U.mean()), 'mean_gpt_final_pct': float(SD.gpt_final_pct.mean()), 'mean_copied_share': float(SD.copied.mean()),
                   'share_ge_half_U': float((SD.U >= .5).mean()), 'share_ge_half_gpt_final_pct': float((SD.gpt_final_pct >= .5).mean()),
                   'share_ge_half_copied': float((SD.copied >= .5).mean()),
                   'agreement_at_half_U_vs_gpt_final_pct': float(((SD.U >= .5) == (SD.gpt_final_pct >= .5)).mean())}
o['N5_naive'] = {'mean_naive': float(SD.naive.mean()), 'share_ge_half_naive': float((SD.naive >= .5).mean())}
o['students_without_replies'] = int((SD.n_replies == 0).sum())
o['per_student'] = stud
json.dump(o, open(os.path.join(HERE, '..', 'results', 'r22_nirvana.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in o.items() if k != 'per_student'}, indent=1, default=float))
