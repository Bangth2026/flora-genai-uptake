"""Analysis log R21, B-46 (benchmark): chatbot text revised by a language model before it is pasted.
Three routes are added to the 13 specified routes at weight 0.05 each (seed 2900). Each takes whole sentences of one of
the borrowed replies (raw reply split at . ! ? : and line breaks, as in R20 B-42; at most 150 tokens) and has an open
instruction-tuned model (Qwen2.5-3B-Instruct, Q4_K_M GGUF, llama.cpp, greedy decoding) revise them as a writer would,
under one of three instructions: light edit, rewrite in the writer's own words, condense to about half. The revised
passage is pasted in one insertion. Planted truth: a token of the revised passage is chatbot text (C) if it lies in a
run of at least four tokens aligned with the reply passage (difflib, no autojunk), otherwise the writer's edit (T).
The model call is made once per (donor, reply, first sentence, instruction) and cached. So that the length of a revised
passage does not change later random draws, the move route draws its insertion point with one call of random()
instead of randrange(); the set is then built twice, once to collect the passages and once with their revisions.
No measure, class definition, parameter or code of the attribution changes.
Usage: python -I src/r21_benchmark.py DATA_DIR R8_TOKEN_CLASSES WORK_DIR GGUF -> results/r21_benchmark.json,
WORK_DIR/r21_llm_cache.json (the revised passages, also copied to results/r21_llm_revisions.json.gz)"""
import sys, os, json, random, time, re, gzip, hashlib
from difflib import SequenceMatcher
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DATA_DIR, R8, WORK, GGUF = sys.argv[1:5]
NS = int(os.environ.get('NSTUD', 200)); SEED = 2900; W = .05; MAXSRC = 150
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)

r15 = open(os.path.join(HERE, 'r15_benchmark.py')).read()
NEW = r15[r15.index('NEW = """') + 9: r15.index('"""\nanchor')] + '\n'
NEW3 = """        elif r.startswith('C-llm'):
            ri = R.randrange(len(REPL[u])); rep = REPL[u][ri]; used.add(ri); ask(R.choice(PROM[u]), rep)
            s0, raw = _pick_raw(R, u, ri, L); blk, lb = _llm_edit(u, ri, s0, r[len('C-llm-'):], raw)
            doc.extend(blk); truth.extend([lab if x else 'Tedit' for x in lb]); snap()
"""
src = open(os.path.join(HERE, 'r11_benchmark.py')).read().replace("'results/flora_token_classes.csv.gz'", repr(R8))
cut = src.index("out = {'n_per_seed': 400}"); anchor = "        elif r.startswith('E'):"
defs = src[:cut].replace(anchor, NEW + NEW3 + anchor)
# the move route draws its target position with randrange(0, start + 1), whose use of the random stream depends on the
# length of the document; a revised passage changes that length, so here the position is drawn with one call of
# random() and later draws no longer depend on the revisions (this set is new and is not compared student by student)
assert defs.count("at = R.randrange(0, start + 1)") == 1
defs = defs.replace("at = R.randrange(0, start + 1)", "at = int(R.random() * (start + 1))")
defs = defs.replace("TRUTHMAP = {'M': 'T',", "TRUTHMAP = {'A3-after': 'X', 'A3-before': 'X', 'A3-regen': 'X', 'M': 'T',")
sys.argv = [os.path.join(HERE, 'r11_benchmark.py'), DATA_DIR]
G_ = {'__name__': 'r11_defs', '__file__': os.path.join(HERE, 'r11_benchmark.py')}
SPLIT = re.compile(r'(?<=[.!?:])\s+|\n+')
CACHE_F = os.path.join(WORK, 'r21_llm_cache.json')
CACHE = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
MISS = {}; USED = set()

def _pick_raw(R, u, ri, L):
    ss = RAWS[u][ri]
    if len(ss) <= 2: s0 = 0; got = list(ss)
    else:
        s0 = R.randrange(len(ss)); got = []; n_ = 0; q = s0
        while q < len(ss) and (n_ < L or len(got) < 2):
            k = len(G_['tokens'](ss[q]))
            if got and n_ + k > MAXSRC: break
            got.append(ss[q]); n_ += k; q += 1
    return s0, ' '.join(got)
def _key(u, ri, s0, style): return f'{u}|{ri}|{s0}|{style}'
def _llm_edit(u, ri, s0, style, raw):
    tk = G_['tokens']; srct = tk(raw)[:MAXSRC]; k = _key(u, ri, s0, style)
    USED.add(k)
    if k in CACHE: out = tk(CACHE[k]['revised'])
    else: MISS[k] = dict(source=raw, style=style); out = list(srct)
    if not out: out = list(srct)
    lab = [False] * len(out)
    for a, b, size in SequenceMatcher(None, srct, out, autojunk=False).get_matching_blocks():
        if size >= 4: lab[b:b + size] = [True] * size
    return out, lab

exec(compile(defs, 'r11_benchmark.py', 'exec'), G_)
G_['_pick_raw'] = _pick_raw; G_['_llm_edit'] = _llm_edit; G_['TRUTHMAP']['Tedit'] = 'T'
RAWS = {}
for u_, g_ in G_['d'].groupby('user_id'):
    RAWS[u_] = [[s.strip() for s in SPLIT.split(str(x)) if G_['tokens'](s)] for x in g_.chatbot_utterance_text.astype(str)]
make_student, BASE, attribute, counts, prf_from, ORDER = (G_[k] for k in ('make_student', 'BASE', 'attribute', 'counts', 'prf_from', 'ORDER'))
LLMR = ['C-llm-light', 'C-llm-rewrite', 'C-llm-condense']
ROUTES = BASE + [(r, W) for r in LLMR]

INSTR = {'light': 'Edit this passage lightly so that it fits into your proposal: change wording you would put differently and cut anything unnecessary, but keep most of it.',
         'rewrite': 'Rewrite this passage in your own words for your proposal, keeping its meaning.',
         'condense': 'Condense this passage to about half its length for your proposal.'}
SYSTEM = 'You are a postgraduate student writing a project proposal for a data-science course. A chatbot gave you the passage below. Return only the revised passage, with no comments.'
LLM = None
def generate(miss):
    global LLM
    if LLM is None:
        from llama_cpp import Llama
        LLM = Llama(model_path=GGUF, n_ctx=1024, n_threads=os.cpu_count(), seed=0, verbose=False)
    for i, (k, v) in enumerate(sorted(miss.items())):
        n_src = len(G_['tokens'](v['source']))
        r = LLM.create_chat_completion(messages=[{'role': 'system', 'content': SYSTEM},
                                                 {'role': 'user', 'content': INSTR[v['style']] + '\n\nPassage:\n' + v['source']}],
                                       temperature=0.0, max_tokens=int(min(1.6 * n_src + 30, 300)))
        CACHE[k] = dict(source=v['source'], style=v['style'], revised=r['choices'][0]['message']['content'].strip())
        if i % 10 == 9 or i == len(miss) - 1:
            json.dump(CACHE, open(CACHE_F, 'w')); log('generated', i + 1, 'of', len(miss))

for it in range(12):
    MISS.clear(); USED.clear(); R = random.Random(SEED); STUD = [make_student(R, ROUTES) for _ in range(NS)]
    log('build', it, 'missing', len(MISS), 'cached', len(CACHE))
    if not MISS: break
    generate(dict(MISS))
assert not MISS

cn, Ut, Ue = [], [], []; byr = {}
for st in STUD:
    cls = attribute(st, st['snaps'], entry_rule='token')['cls']; cn.append(counts(st['truth'], cls))
    Ut.append(st['truth'].count('C') / len(cls)); Ue.append(cls.count('C') / len(cls))
    prev = None
    for r, c, t in zip(st['route'], cls, st['truth']):
        key = r if r != 'Tedit' else (prev + ':edit' if prev else 'Tedit')
        if r != 'Tedit': prev = r if r.startswith('C-llm') else None
        byr.setdefault(key, {}); byr[key][c] = byr[key].get(c, 0) + 1
out = {'n_students': NS, 'seed': SEED, 'route_weight': W, 'model': os.path.basename(GGUF)}
out['classes'] = {c: dict(zip(('precision', 'recall', 'f1'), prf_from(cn, c))) for c in ORDER}
Ut, Ue = np.array(Ut), np.array(Ue)
out['U_error'] = {'mean_signed': float((Ue - Ut).mean()), 'mae': float(np.abs(Ue - Ut).mean()), 'true_mean_C': float(Ut.mean()),
                  'agreement_at_half': float(np.mean((Ut >= .5) == (Ue >= .5)))}
out['route_to_class'] = {r: {c: v / sum(d_.values()) for c, v in d_.items()} for r, d_ in byr.items()
                         if r.startswith('C-') or r.endswith(':edit')}
# per instruction (passages used in the set): retention of reply text in the revised passage and recall of the surviving reply text
used = {}
for st in STUD:
    for r in set(st['route']):
        if r.startswith('C-llm'): used[r] = used.get(r, 0) + 1
ret = {}
for k, v in CACHE.items():
    if k not in USED: continue
    s = G_['tokens'](v['source'])[:MAXSRC]; o = G_['tokens'](v['revised'])
    if not o: continue
    m4 = sum(sz for a, b, sz in SequenceMatcher(None, s, o, autojunk=False).get_matching_blocks() if sz >= 4)
    m8 = sum(sz for a, b, sz in SequenceMatcher(None, s, o, autojunk=False).get_matching_blocks() if sz >= 8)
    ret.setdefault(v['style'], []).append((len(s), len(o), m4 / len(o), m8 / len(o)))
out['revisions'] = {st_: {'n': len(x), 'mean_source_tokens': float(np.mean([a for a, *_ in x])), 'mean_revised_tokens': float(np.mean([b for _, b, *_ in x])),
                          'share_in_runs_ge4': float(np.mean([c for *_, c, _ in x])), 'share_in_runs_ge8': float(np.mean([d for *_, d in x])),
                          'share_of_passages_with_no_run_ge8': float(np.mean([d == 0 for *_, d in x]))} for st_, x in ret.items()}
out['students_with_route'] = used
out['seconds'] = time.perf_counter() - t00
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r21_benchmark.json'), 'w'), indent=1, default=float)
out['n_revisions_used'] = len(USED); out['n_revisions_generated'] = len(CACHE)
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r21_benchmark.json'), 'w'), indent=1, default=float)
with gzip.open(os.path.join(HERE, '..', 'results', 'r21_llm_revisions.json.gz'), 'wt') as f: json.dump({k: v for k, v in CACHE.items() if k in USED}, f)
log('done', out['classes']['C'], out['U_error'])
