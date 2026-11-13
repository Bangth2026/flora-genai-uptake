"""Analysis log R11, B-7 (i): LLM baseline on the planted-provenance benchmark.

For each passage of 20 tokens of a simulated student's final text, the model receives the passage, the reply passage
with the largest 8-gram overlap and the student turn with the largest overlap (each with its time), and the time the
passage first appeared in the editor, and answers CHATBOT, STUDENT or NEITHER. Tokens of passages answered CHATBOT are
marked as chatbot text and scored against the planted class C, as for the other baselines.

Two back ends:
  local  (default): a Hugging Face model run on CPU, greedy decoding (MODEL, default Qwen/Qwen2.5-1.5B-Instruct)
  api    : any OpenAI-compatible chat endpoint; set LLM_BACKEND=api, API_BASE, API_KEY and MODEL
Usage: python -I src/r11_llm_baseline.py WORK_DIR [N_STUDENTS] -> results/r11_llm_baseline.json
WORK_DIR holds r11_students.pkl written by src/r11_benchmark.py (seed 2100, R10 routes)."""
import sys, os, json, pickle, time, re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demo'))
from provenance import tokens, ngrams

WORK = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
BACKEND = os.environ.get('LLM_BACKEND', 'local'); MODEL = os.environ.get('MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
CH, N = 20, 8
STUD = pickle.load(open(f'{WORK}/r11_students_llm.pkl' if os.path.exists(f'{WORK}/r11_students_llm.pkl') else f'{WORK}/r11_students.pkl', 'rb'))['base'][:NS]

SYSTEM = ("You audit how text entered a student's document in a writing environment with a chatbot. "
          "Decide who first produced the passage. Answer with one word: CHATBOT if the passage was taken from a chatbot reply, "
          "STUDENT if the student wrote it (including text the student wrote first and the chatbot then repeated), "
          "NEITHER if it came from somewhere else.")

def window(toks, hit, w=30):
    return ' '.join(toks[max(0, hit - w):hit + w])

def best(items, chunk):
    cg = set(ngrams(chunk, N)); top = (0, None, None, None)
    for t, x in items:
        tk = tokens(x)
        for i, g in enumerate(ngrams(tk, N)):
            if g in cg:
                k = sum(1 for h in ngrams(tk, N) if h in cg)
                if k > top[0]: top = (k, t, tk, i)
                break
    return top

def first_editor_time(snaps, chunk):
    g = ' '.join(chunk[:N])
    for t, x in snaps:
        if g in ' '.join(tokens(x)): return t
    return None

def prompt_for(st, chunk):
    kr, tr_, rk, ri = best(st['replies'], chunk); kp, tp_, pk, pi = best(st['turns'], chunk); te = first_editor_time(st['snaps'], chunk)
    f = lambda t: 'unknown' if t is None else f'{t:.0f} s'
    rep = f'Chatbot reply at {f(tr_)}: "{window(rk, ri)}"' if rk else 'No chatbot reply contains this passage.'
    pro = f'Student message at {f(tp_)}: "{window(pk, pi)}"' if pk else 'No student message contains this passage.'
    return (f'Passage (first appeared in the editor at {f(te)}): "{" ".join(chunk)}"\n{rep}\n{pro}\n'
            'Times are seconds on one clock. Who first produced the passage? Answer CHATBOT, STUDENT or NEITHER.')

if BACKEND == 'local':
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(os.cpu_count() or 1)
    tkz = AutoTokenizer.from_pretrained(MODEL); mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16); mdl.eval()
    def ask(p):
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': p}]
        ids = tkz.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True)['input_ids']
        with torch.no_grad(): out = mdl.generate(ids, max_new_tokens=4, do_sample=False)
        return tkz.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
else:
    import urllib.request
    def ask(p):
        body = json.dumps({'model': MODEL, 'temperature': 0, 'max_tokens': 4,
                           'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': p}]}).encode()
        req = urllib.request.Request(os.environ['API_BASE'].rstrip('/') + '/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['API_KEY']})
        return json.load(urllib.request.urlopen(req, timeout=120))['choices'][0]['message']['content']

t0 = time.perf_counter(); tp = pp = ap = 0; per = []; answers = {'CHATBOT': 0, 'STUDENT': 0, 'NEITHER': 0, 'other': 0}; calls = 0
for k, st in enumerate(STUD):
    A = tokens(st['final']); mark = [False] * len(A)
    for s in range(0, len(A), CH):
        chunk = A[s:s + CH]
        if len(chunk) < N: continue
        a = re.sub(r'[^A-Z]', '', ask(prompt_for(st, chunk)).upper()); calls += 1
        lab = 'CHATBOT' if a.startswith('CHATBOT') else 'STUDENT' if a.startswith('STUDENT') else 'NEITHER' if a.startswith('NEITHER') else 'other'
        answers[lab] += 1
        if lab == 'CHATBOT':
            for j in range(s, min(len(A), s + CH)): mark[j] = True
    truthC = [x == 'C' for x in st['truth']]
    a_, b_, c_ = sum(x and y for x, y in zip(truthC, mark)), sum(mark), sum(truthC); tp += a_; pp += b_; ap += c_; per.append((a_, b_, c_))
    print(k, a_, b_, c_, round(time.perf_counter() - t0), flush=True)
p = tp / pp if pp else float('nan'); r = tp / ap if ap else float('nan'); f1 = 2 * p * r / (p + r) if p + r else float('nan')
rng = np.random.default_rng(2130); bs = []
for _ in range(2000):
    idx = rng.integers(0, len(per), len(per)); x = [per[i] for i in idx]
    P_ = sum(a for a, _, _ in x) / max(1, sum(b for _, b, _ in x)); R_ = sum(a for a, _, _ in x) / max(1, sum(c for _, _, c in x))
    bs.append(2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0)
json.dump({'backend': BACKEND, 'model': MODEL, 'n_students': len(STUD), 'calls': calls, 'passage_tokens': CH, 'answers': answers,
           'token_chatbot_text': {'precision': p, 'recall': r, 'f1': f1, 'ci_f1': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
           'per_student': per, 'seconds': time.perf_counter() - t0}, open('results/r11_llm_baseline.json', 'w'), indent=1)
print('done', p, r, f1)
