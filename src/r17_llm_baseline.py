"""Analysis log R17, B-35: LLM baseline on three benchmark sets (the first NSTUD simulated students, default 20, of each of the sets with the 13
specified routes, with all 20 routes, and with the assumption-A3 routes; written by src/r17_benchmark.py, B-34).
The prompt, passage length and scoring are those of src/r11_llm_baseline.py: for each passage of 20 tokens the model
receives the passage with its first editor time, the reply passage with the largest 8-gram overlap and the writer
turn with the largest overlap (each with its time), and answers CHATBOT, STUDENT or NEITHER.
Back ends: gguf (default; a quantised instruction model run on CPU with llama.cpp, greedy decoding; MODEL_PATH) or api
(any OpenAI-compatible endpoint; LLM_BACKEND=api, API_BASE, API_KEY, MODEL). The first three students of the first set
are run twice to check that answers repeat.
Usage: python -I src/r17_llm_baseline.py WORK_DIR -> results/r17_llm_baseline.json"""
import sys, os, json, pickle, time, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'r11_llm_baseline.py')).read()
pre = src[src.index('SYSTEM = ('): src.index("if BACKEND == 'local':")]
sys.path.insert(0, os.path.join(HERE, '..', 'demo'))
from provenance import tokens, ngrams
N, CH = 8, 20
exec(pre)   # SYSTEM, window, best, first_editor_time, prompt_for
WORK = sys.argv[1]; SETS = pickle.load(open(f'{WORK}/r17_llm_students.pkl', 'rb'))
NSTUD = int(os.environ.get('NSTUD', 20)); SETS = {k: v[:NSTUD] for k, v in SETS.items()}
BACKEND = os.environ.get('LLM_BACKEND', 'gguf')
if BACKEND == 'gguf':
    from llama_cpp import Llama
    MP = os.environ['MODEL_PATH']; MODEL = os.path.basename(MP)
    llm = Llama(model_path=MP, n_ctx=2048, n_threads=int(os.environ.get('THREADS', 2)), verbose=False, seed=0)
    def ask(p):
        r = llm.create_chat_completion(messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': p}], max_tokens=4, temperature=0)
        return r['choices'][0]['message']['content']
else:
    import urllib.request
    MODEL = os.environ['MODEL']
    def ask(p):
        body = json.dumps({'model': MODEL, 'temperature': 0, 'max_tokens': 4,
                           'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': p}]}).encode()
        req = urllib.request.Request(os.environ['API_BASE'].rstrip('/') + '/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['API_KEY']})
        return json.load(urllib.request.urlopen(req, timeout=120))['choices'][0]['message']['content']

def run(STUD, keep=False):
    per, answers, calls, seq = [], {'CHATBOT': 0, 'STUDENT': 0, 'NEITHER': 0, 'other': 0}, 0, []
    for st in STUD:
        A = tokens(st['final']); mark = [False] * len(A)
        for s in range(0, len(A), CH):
            chunk = A[s:s + CH]
            if len(chunk) < N: continue
            a = re.sub(r'[^A-Z]', '', ask(prompt_for(st, chunk)).upper()); calls += 1
            lab = 'CHATBOT' if a.startswith('CHATBOT') else 'STUDENT' if a.startswith('STUDENT') else 'NEITHER' if a.startswith('NEITHER') else 'other'
            answers[lab] += 1; seq.append(lab)
            if lab == 'CHATBOT':
                for j in range(s, min(len(A), s + CH)): mark[j] = True
        tC = [x == 'C' for x in st['truth']]
        per.append((sum(x and y for x, y in zip(tC, mark)), sum(mark), sum(tC)))
    return per, answers, calls, seq

def prf(per, rng):
    s = np.array(per, float).sum(0); p = s[0] / s[1] if s[1] else float('nan'); r = s[0] / s[2]; f = 2 * p * r / (p + r) if p + r else float('nan'); bs = []
    a = np.array(per, float)
    for _ in range(2000):
        x = a[rng.integers(0, len(a), len(a))].sum(0); P_ = x[0] / max(x[1], 1); R_ = x[0] / max(x[2], 1); bs.append(2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0)
    return {'precision': float(p), 'recall': float(r), 'f1': float(f), 'f1_ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

t0 = time.perf_counter(); rng = np.random.default_rng(2175); out = {'backend': BACKEND, 'model': MODEL, 'passage_tokens': CH, 'sets': {}}
for name in ('base', 'ext', 'a3'):
    t1 = time.perf_counter(); per, ans, calls, seq = run(SETS[name])
    out['sets'][name] = {'n_students': len(SETS[name]), 'calls': calls, 'answers': ans, 'token_chatbot_text': prf(per, rng), 'per_student': per,
                         'seconds': time.perf_counter() - t1}
    if name == 'base': out['_seq_base3'] = seq[:sum(1 for st in SETS['base'][:3] for s in range(0, len(tokens(st['final'])), CH) if len(tokens(st['final'])[s:s + CH]) >= N)]
    print(name, out['sets'][name]['token_chatbot_text'], ans, round(time.perf_counter() - t0), flush=True)
    json.dump(out, open(os.path.join(HERE, '..', 'results', 'r17_llm_baseline.json'), 'w'), indent=1)
_, _, _, seq2 = run(SETS['base'][:3])
out['repeat_first_three_students_identical_answers'] = float(np.mean([a == b for a, b in zip(out.pop('_seq_base3'), seq2)]))
out['seconds'] = time.perf_counter() - t0
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r17_llm_baseline.json'), 'w'), indent=1)
print('done', out['repeat_first_three_students_identical_answers'])
