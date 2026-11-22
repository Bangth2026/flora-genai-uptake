"""Analysis log R22, N-6 to N-9 (post hoc, after the results of N-1 to N-5 were seen).
N-6  tolerant matching on NIRVANA: src/r22_nirvana.py executed with reply and prompt n-grams matched exactly or with one
     token replaced (as R15), against the same copy signal.
N-7  C tokens without a ChatGPT copy event: by entry (typed, paste without a matching copy, move within the editor), run
     length, and, for typed runs, whether their 8-grams also occur in replies given to other NIRVANA students (wording
     the model produces for others) and the time from the first reply containing them to their entry.
N-8  X tokens by entry, and the distribution of tokens inserted per editor insert operation (the editor logs some typing in
     chunks of several words).
N-9  the class shares and student-level agreement with tau chosen from NIRVANA's own logs (the 99th percentile of tokens
     per insert operation), as the method's documentation asks for new logs; reported next to the frozen tau = 5.
Usage: python -I src/r22_nirvana_posthoc.py NIRVANA_XLSX -> results/r22_nirvana_posthoc.json"""
import sys, os, re, json, ast, io, contextlib
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); XL = sys.argv[1]
src = open(os.path.join(HERE, 'r22_nirvana.py')).read()
def run_variant(tolerant=False, tau=5):
    s = src.replace("results', 'r22_nirvana.json'", "results', '_tmp_r22.json'").replace("results', 'r22_nirvana_tokens.csv.gz'", "results', '_tmp_r22_tokens.csv.gz'")
    s = s.replace("N, TAU = 8, 5", f"N, TAU = 8, {tau}")
    if tolerant:
        s = s.replace("def dialogue(u):", '''class Fuzzy(dict):
    def __init__(self, items, n):
        super().__init__()
        for t, text in sorted(items, key=lambda x: x[0]):
            for g in grams(tok(text), n):
                for key in [g] + [g[:i] + ('\\0',) + g[i + 1:] for i in range(n)]: dict.setdefault(self, key, t)
    def _keys(self, g): return [g] + [g[:i] + ('\\0',) + g[i + 1:] for i in range(len(g))]
    def __contains__(self, g): return any(dict.__contains__(self, k) for k in self._keys(g))
    def __getitem__(self, g): return min(dict.__getitem__(self, k) for k in self._keys(g) if dict.__contains__(self, k))
def dialogue_exact(u):''')
        s = s.replace("    tbot, tpr = dialogue(u); FG", "    x_ = e[e.essay_num == u]; tbot = Fuzzy(list(zip(x_[x_.op_type == 'gpt_response'].time.astype(float), x_[x_.op_type == 'gpt_response'].selected_text)), N); tpr = Fuzzy(list(zip(x_[x_.op_type == 'gpt_inquiry'].time.astype(float), x_[x_.op_type == 'gpt_inquiry'].selected_text)), N); FG")
    sys.argv = [os.path.join(HERE, 'r22_nirvana.py'), XL]
    G = {'__name__': 'r22v', '__file__': os.path.join(HERE, 'r22_nirvana.py')}
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(s, 'r22_nirvana.py', 'exec'), G)
    o = json.load(open(os.path.join(HERE, '..', 'results', '_tmp_r22.json'))); T = pd.read_csv(os.path.join(HERE, '..', 'results', '_tmp_r22_tokens.csv.gz'))
    os.remove(os.path.join(HERE, '..', 'results', '_tmp_r22.json')); os.remove(os.path.join(HERE, '..', 'results', '_tmp_r22_tokens.csv.gz'))
    keep = ('class_shares_mean', 'N1_C_tokens_entered_by_chatgpt_copy', 'N1_C_tokens_typed', 'N2_chatgpt_copied_tokens_by_class',
            'N3_E_tokens_chatgpt_copied', 'N3_X_tokens_chatgpt_copied', 'N3_T_tokens_chatgpt_copied', 'N4_student')
    return {k: o[k] for k in keep}, T
out = {}
out['N6_tolerant'], _ = run_variant(tolerant=True)
# N-7, N-8 on the frozen run
T = pd.read_csv(os.path.join(HERE, '..', 'results', 'r22_nirvana_tokens.csv.gz'))
e = pd.read_excel(XL, sheet_name='Essay Data').sort_values(['essay_num', 'op_index'])
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower()); grams = lambda t, k: [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
C = T[T.cls == 'C']; out['N7_C_by_entry'] = C.entry.value_counts(normalize=True).to_dict()
def runs(df, mask):
    res = []
    for u, g in df.groupby('user'):
        m = mask.loc[g.index].values; i = 0
        while i < len(m):
            if m[i]:
                j = i
                while j < len(m) and m[j]: j += 1
                res.append((u, g.token_index.values[i], j - i)); i = j
            else: i += 1
    return res
tr = runs(T, (T.cls == 'C') & (T.entry == 'typed'))
L = np.array([r[2] for r in tr])
other_reply = {}
for u, g in e[e.op_type == 'gpt_response'].groupby('essay_num'):
    for s in g.selected_text: 
        for gg in set(grams(tok(s), 8)): other_reply.setdefault(gg, set()).add(u)
gen = 0; gen_tok = 0
for u, i0, ln in tr:
    toks = T[(T.user == u)].token.astype(str).tolist()[i0:i0 + ln]
    gs = grams(toks, 8) or [tuple(toks)]
    if all(len(other_reply.get(g, set()) - {u}) > 0 for g in gs): gen += 1; gen_tok += ln
out['N7_typed_C_runs'] = {'runs': len(tr), 'tokens': int(L.sum()), 'students': len({r[0] for r in tr}), 'median_run': float(np.median(L)),
                          'share_tokens_in_runs_ge_16': float(L[L >= 16].sum() / L.sum()), 'share_tokens_in_runs_lt_12': float(L[L < 12].sum() / L.sum()),
                          'runs_all_8grams_in_other_students_replies': gen, 'tokens_in_those_runs': int(gen_tok)}
pu = T[(T.cls == 'C') & (T.entry == 'paste:unmatched')]
out['N7_C_unmatched_paste'] = {'tokens': int(len(pu)), 'students': int(pu.user.nunique())}
out['N8_X_by_entry'] = T[T.cls == 'X'].entry.value_counts(normalize=True).to_dict()
sz = []
for u, g in e[(e.op_loc == 'editor') & e.current_editor.notna()].groupby('essay_num'):
    prev = []
    for ot, v in zip(g.op_type, g.current_editor):
        try: cur = tok('\n'.join(ast.literal_eval(v)))
        except Exception: continue
        if ot == 'i': sz.append(max(0, len(cur) - len(prev)))
        prev = cur
sz = np.array(sz)
out['N8_insert_ops'] = {'n': int(len(sz)), 'share_gt_5_tokens': float((sz > 5).mean()), 'tokens_from_ops_gt_5': int(sz[sz > 5].sum()),
                        'p50_p90_p99_of_ops_gt_0': [float(x) for x in np.percentile(sz[sz > 0], [50, 90, 99])]}
tau99 = int(np.percentile(sz[sz > 0], 99)); out['N9_tau_from_logs'] = tau99
out['N9_variant'], _ = run_variant(tau=tau99)
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r22_nirvana_posthoc.json'), 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
