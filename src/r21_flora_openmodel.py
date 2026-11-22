"""Analysis log R21, F-49: verbatim overlap between FLoRA proposals and an open model's replies to the students' own
prompts (reviewer question on the rate of A3 violations). For 50 students drawn at random (seed 2211), up to four of
each student's prompts (random, same seed) are sent to Qwen2.5-1.5B-Instruct (Q4_K_M GGUF, llama.cpp, greedy decoding,
at most 256 new tokens, a neutral system prompt; no dialogue history). For each student: the share of the proposal's
tokens of each released class (C, E, S, T, X) covered by an 8-gram of these replies that occurs in none of the
student's prompts (so that a reply repeating the writer's draft does not count); the same with the replies the
open model gave to another sampled student's prompts (chance control, the next student in the sample); and the share
of the proposal covered by the GPT-4o replies logged for the same prompts. Text covered only by 8-grams that occur in
at least five other proposals is reported separately. The open model is not the embedded GPT-4o, so this estimates how
much of each class is wording that a different chatbot also produces verbatim for the same request, not the A3 rate
itself. Usage: python -I src/r21_flora_openmodel.py DATA_DIR WORK_DIR GGUF -> results/r21_flora_openmodel.json,
WORK_DIR/r21_openmodel_replies.json (the generated replies, also copied to results/)"""
import sys, os, re, json, time, random, gzip
from collections import Counter
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); D, WORK, GGUF = sys.argv[1:4]
NSAMP, NPR, N = int(os.environ.get('NSAMP', 50)), 4, 8
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t: [tuple(t[i:i + N]) for i in range(len(t) - N + 1)]
tc = pd.read_csv(os.path.join(HERE, '..', 'results', 'flora_token_classes.csv.gz')).sort_values(['user_id', 'token_index'])
d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
users = sorted(set(tc.user_id) & set(d.user_id)); R = random.Random(2211); samp = R.sample(users, NSAMP)
A = {u: g.token.astype(str).tolist() for u, g in tc.groupby('user_id')}; CL = {u: g['class'].tolist() for u, g in tc.groupby('user_id')}
peer = Counter(g for u in A for g in set(grams(A[u])))
req = {}
for u in samp:
    x = d[d.user_id == u].reset_index(drop=True); idx = sorted(R.sample(range(len(x)), min(NPR, len(x))))
    req[u] = [(str(x.user_utterance_text[i]), str(x.chatbot_utterance_text[i])) for i in idx]
CF = os.path.join(WORK, 'r21_openmodel_replies.json'); cache = json.load(open(CF)) if os.path.exists(CF) else {}
SYSTEM = 'You are a helpful assistant supporting a postgraduate student who is writing a project proposal in a data-science course.'
todo = [(u, k, p) for u in samp for k, (p, _) in enumerate(req[u]) if f'{u}|{k}' not in cache]
if todo:
    from llama_cpp import Llama
    LLM = Llama(model_path=GGUF, n_ctx=2048, n_threads=os.cpu_count(), seed=0, verbose=False)
    for i, (u, k, p) in enumerate(todo):
        r = LLM.create_chat_completion(messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': p[:3000]}], temperature=0.0, max_tokens=256)
        cache[f'{u}|{k}'] = r['choices'][0]['message']['content']
        if i % 10 == 9 or i == len(todo) - 1: json.dump(cache, open(CF, 'w')); log('generated', i + 1, 'of', len(todo))
def cover(Atok, G):
    m = np.zeros(len(Atok), bool)
    for i, g in enumerate(grams(Atok)):
        if g in G: m[i:i + N] = True
    return m
rows = []
for j, u in enumerate(samp):
    v = samp[(j + 1) % len(samp)]
    Pu = {g for p in d[d.user_id == u].user_utterance_text for g in grams(tok(p))}; Pv = {g for p in d[d.user_id == v].user_utterance_text for g in grams(tok(p))}
    Go = {g for k in range(len(req[u])) for g in grams(tok(cache[f'{u}|{k}']))} - Pu
    Gv = {g for k in range(len(req[v])) for g in grams(tok(cache[f'{v}|{k}']))} - Pv
    G4 = {g for _, rep in req[u] for g in grams(tok(rep))}
    Gs = {g for g in grams(A[u]) if peer[g] - 1 >= 5}
    mo, mv, m4 = cover(A[u], Go), cover(A[u], Gv), cover(A[u], G4)
    mo_ns = cover(A[u], Go - Gs)
    c = np.array(CL[u]); row = {'user': int(u), 'tokens': len(c), 'open_all': mo.mean(), 'open_all_outside_shared': mo_ns.mean(), 'control_all': mv.mean(), 'gpt4o_same_prompts_all': m4.mean(),
                                'open_overlap_with_gpt4o_reply_8grams': len(Go & G4) / max(len(Go), 1)}
    for k in 'CESTX':
        s = c == k; row[f'n_{k}'] = int(s.sum())
        row[f'open_{k}'] = float(mo[s].mean()) if s.any() else np.nan; row[f'control_{k}'] = float(mv[s].mean()) if s.any() else np.nan
        row[f'open_{k}_tokens'] = int(mo[s].sum()); row[f'control_{k}_tokens'] = int(mv[s].sum())
    rows.append(row)
df = pd.DataFrame(rows)
out = {'n_students': NSAMP, 'prompts_per_student': NPR, 'n_prompts': int(sum(len(v) for v in req.values())), 'model': os.path.basename(GGUF),
       'mean_open_reply_tokens': float(np.mean([len(tok(v)) for v in cache.values()]))}
out['mean_share_covered'] = {k: float(df[k].mean()) for k in ('open_all', 'open_all_outside_shared', 'control_all', 'gpt4o_same_prompts_all', 'open_overlap_with_gpt4o_reply_8grams')}
out['pooled_share_of_class_covered'] = {k: {'open': df[f'open_{k}_tokens'].sum() / max(df[f'n_{k}'].sum(), 1), 'control': df[f'control_{k}_tokens'].sum() / max(df[f'n_{k}'].sum(), 1),
                                            'class_tokens': int(df[f'n_{k}'].sum())} for k in 'CESTX'}
out['students_with_any_open_coverage'] = float((df.open_all > 0).mean())
out['students_with_open_coverage_of_X'] = float((df.open_X_tokens > 0).mean())
out['per_student'] = rows
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r21_flora_openmodel.json'), 'w'), indent=1, default=float)
with gzip.open(os.path.join(HERE, '..', 'results', 'r21_openmodel_replies.json.gz'), 'wt') as f: json.dump(cache, f)
log('done', out['mean_share_covered'], out['pooled_share_of_class_covered'])
