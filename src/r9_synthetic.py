"""Analysis log R9, S-1: accuracy of the five-class attribution against known routes in simulated logs.

Each simulated student writes a document from blocks whose route of entry is known:
  T  typed word by word in the editor
  C  copied from a chatbot reply (pasted as a block, or retyped word by word)
  E1 the student's own draft (from outside the logs) sent in a prompt, repeated by the reply, then pasted from the reply
  E2 typed in the editor, later sent to the chatbot, which repeats it (nothing pasted back)
  E3 typed, deleted, sent to the chatbot and pasted back from its reply
  S  text from a shared source (pasted)
  X  pasted from a source outside the logs
The attribution of demo/provenance.py (the rules of the article) is applied unchanged, and token-level precision and
recall are computed per class under perturbations: clock offset between dialogue and editor, snapshot thinning,
partial paraphrase of pasted chatbot text. Ground truth for E1, E2, E3 is E (the text entered through the student).
Usage: python -I src/r9_synthetic.py -> results/r9_synthetic.json"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demo'))
from provenance import attribute, tokens

R = random.Random(2090)
SYL = [a + b for a in 'bcdfghjklmnprstvwz' for b in ['a', 'e', 'i', 'o', 'u', 'ai', 'ou', 'ea']]
VOCAB = sorted({''.join(R.choice(SYL) for _ in range(R.randint(1, 3))) for _ in range(6000)})
FUNC = 'the of and to in a is that for on with as by this are be from at or an it'.split()
def words(k):  # prose-like: content words mixed with frequent function words
    return [R.choice(FUNC) if R.random() < 0.35 else R.choice(VOCAB) for _ in range(k)]
SHARED = [words(40) for _ in range(6)]

def student(cfg):
    t = 0.0; doc = []; snaps = []; replies = []; turns = []; truth = []
    def snap(): snaps.append((t, ' '.join(doc)))
    routes = ['T', 'T', 'C', 'Cr', 'E1', 'E2', 'E3', 'S', 'X', 'X']
    R.shuffle(routes)
    for r in routes[:R.randint(5, 10)]:
        k = R.randint(20, 60); blk = words(k); t += R.uniform(30, 600)
        if r == 'T':
            for wd in blk: t += R.uniform(0.2, 1.5); doc.append(wd); truth.append('T'); snap()
        elif r in ('C', 'Cr'):
            turns.append((t, ' '.join(words(8)))); t += 5; replies.append((t, ' '.join(words(5) + blk + words(5)))); t += R.uniform(5, 120)
            if r == 'C':
                if cfg['para']:
                    blk = [R.choice(VOCAB) if (i + 1) % cfg['para'] == 0 else wd for i, wd in enumerate(blk)]
                doc += blk; truth += ['C'] * k; snap()
            else:
                for wd in blk: t += R.uniform(0.3, 1.5); doc.append(wd); truth.append('C'); snap()
        elif r == 'E1':
            turns.append((t, 'improve this ' + ' '.join(blk))); t += 5
            replies.append((t, 'here is a revision ' + ' '.join(blk) + ' ' + ' '.join(words(6)))); t += R.uniform(5, 120)
            doc += blk; truth += ['E'] * k; snap()
        elif r == 'E2':
            for wd in blk: t += R.uniform(0.2, 1.5); doc.append(wd); truth.append('E'); snap()
            t += R.uniform(30, 300); turns.append((t, 'check this ' + ' '.join(blk))); t += 5
            replies.append((t, 'your text ' + ' '.join(blk) + ' reads well'))
        elif r == 'E3':
            start = len(doc)
            for wd in blk: t += R.uniform(0.2, 1.5); doc.append(wd); snap()
            t += 10; del doc[start:]; snap()
            t += R.uniform(10, 120); turns.append((t, 'fix this ' + ' '.join(blk))); t += 5
            replies.append((t, 'fixed ' + ' '.join(blk))); t += R.uniform(5, 60)
            doc += blk; truth += ['E'] * k; snap()
        elif r == 'S':
            blk = list(R.choice(SHARED)); doc += blk; truth += ['S'] * len(blk); snap()
        else:
            doc += blk; truth += ['X'] * k; snap()
    off = cfg['offset']
    replies = [(a + off, b) for a, b in replies]; turns = [(a + off, b) for a, b in turns]
    if cfg['thin']:
        kept, last = [], -1e9
        for s in snaps:
            if s[0] - last >= cfg['thin']: kept.append(s); last = s[0]
        if kept[-1] != snaps[-1]: kept.append(snaps[-1])
        snaps = kept
    final = ' '.join(doc)
    A, cls, _, _ = attribute(final, replies, turns, snaps, shared=[' '.join(s) for s in SHARED], n=8, tau=cfg['tau'], delta=0)
    assert len(A) == len(truth)
    return truth, cls

def evaluate(cfg, n=300):
    pairs = []
    for _ in range(n):
        tr, cl = student(cfg); pairs += list(zip(tr, cl))
    res = {'tokens': len(pairs), 'accuracy': sum(a == b for a, b in pairs) / len(pairs)}
    for c in 'CESTX':
        tp = sum(a == c and b == c for a, b in pairs); pp = sum(b == c for a, b in pairs); ap = sum(a == c for a, b in pairs)
        res[c] = {'precision': tp / pp if pp else None, 'recall': tp / ap if ap else None, 'true_share': ap / len(pairs)}
    conf = {}
    for a, b in pairs: conf.setdefault(a, {}).setdefault(b, 0); conf[a][b] += 1
    res['confusion'] = conf
    return res

base = dict(offset=0, thin=0, para=0, tau=5)
conds = {'baseline': base}
for o in (10, 60, 3600, -60, -3600): conds[f'clock_offset_{o}s'] = {**base, 'offset': o}
for th in (5, 15, 60, 300):
    for tau in (5, 20): conds[f'thin_{th}s_tau{tau}'] = {**base, 'thin': th, 'tau': tau}
for k in (12, 8, 4): conds[f'paraphrase_every_{k}th'] = {**base, 'para': k}
out = {}
for name, cfg in conds.items():
    out[name] = evaluate(cfg); print(name, round(out[name]['accuracy'], 3), {c: (round(out[name][c]['precision'] or 0, 3), round(out[name][c]['recall'] or 0, 3)) for c in 'CESTX'}, flush=True)
json.dump(out, open('results/r9_synthetic.json', 'w'), indent=1)
