"""Analysis log R13, F-13: how much of each provenance class of the FLoRA proposals also occurs verbatim in English
Wikipedia, as a bound on the part of C that could be published web text which the chatbot repeated (assumption A3).

Every 8-gram that lies wholly within one class of a proposal (results/flora_token_classes.csv.gz, token entry rule)
is searched in the plain text of English Wikipedia (wikimedia/wikipedia, dump 20231101.en, 41 Parquet shards on the
Hugging Face hub), tokenised as in the attribution (lower case, alphanumeric runs). A token counts as web-matched
if an 8-gram of its class that covers it occurs in Wikipedia. Wikipedia is one public source among many, so the
shares are lower bounds on web text in general and upper references only for text that Wikipedia contains.
Usage: python -I src/r13_wiki.py SHARD_DIR -> results/r13_wiki.json
SHARD_DIR receives each shard in turn (downloaded, scanned, deleted); WIKI_SHARDS=k limits the run to k shards."""
import sys, os, re, json, time, subprocess
import numpy as np, pandas as pd
import ahocorasick, pyarrow.parquet as pq

SD = sys.argv[1]; os.makedirs(SD, exist_ok=True); N = 8
K = int(os.environ.get('WIKI_SHARDS', 41))
URL = 'https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.en/train/{}.parquet'
tc = pd.read_csv('results/flora_token_classes.csv.gz').sort_values(['user_id', 'token_index'])
NONAN = re.compile(r'[^a-z0-9]+')
win = {}   # pattern -> list of (user, start)
TOKS, CLS = {}, {}
for u, g in tc.groupby('user_id'):
    t = g.token.astype(str).tolist(); c = g['class'].tolist(); TOKS[u], CLS[u] = t, c
    for i in range(len(t) - N + 1):
        if len(set(c[i:i + N])) == 1:
            win.setdefault(' ' + ' '.join(t[i:i + N]) + ' ', []).append((u, i))
A = ahocorasick.Automaton()
for k in win: A.add_word(k, k)
A.make_automaton()
found = set(); t0 = time.perf_counter(); n_art = 0; n_chars = 0
for s in range(K):
    f = os.path.join(SD, f'{s}.parquet')
    if not os.path.exists(f):
        subprocess.run(['curl', '-sSL', '--retry', '3', '-m', '900', '-o', f, URL.format(s)], check=True)
    pf = pq.ParquetFile(f)
    for b in pf.iter_batches(columns=['text'], batch_size=2000):
        txt = ' ' + NONAN.sub(' ', '\n'.join(b.column(0).to_pylist()).lower()) + ' '
        n_art += b.num_rows; n_chars += len(txt)
        for _, k in A.iter(txt): found.add(k)
    os.remove(f)
    print(f'shard {s} done; articles {n_art}; matched windows {len(found)}; {time.perf_counter() - t0:.0f} s', flush=True)
# token-level shares by class
pooled = {c: [0, 0] for c in 'CESTX'}; per = {c: [] for c in 'CESTX'}
for u in TOKS:
    c = CLS[u]; m = np.zeros(len(c), bool)
    for i in range(len(c) - N + 1):
        if len(set(c[i:i + N])) == 1 and (' ' + ' '.join(TOKS[u][i:i + N]) + ' ') in found: m[i:i + N] = True
    for cl in 'CESTX':
        sel = np.array([x == cl for x in c])
        if sel.sum():
            pooled[cl][0] += int((m & sel).sum()); pooled[cl][1] += int(sel.sum()); per[cl].append(float((m & sel).sum() / sel.sum()))
    CLS[u] = (c, m)
mean_prop = {cl: float(np.mean([(CLS[u][1] & np.array([x == cl for x in CLS[u][0]])).sum() / len(CLS[u][0]) for u in TOKS])) for cl in 'CESTX'}
out = {'dump': 'wikimedia/wikipedia 20231101.en', 'shards_scanned': K, 'articles_scanned': n_art, 'characters_scanned': n_chars,
       'n_windows': len(win), 'n_windows_found': len(found),
       'pooled_share_of_class_tokens_web_matched': {cl: (a / b if b else None) for cl, (a, b) in pooled.items()},
       'median_student_share_of_class_tokens_web_matched': {cl: (float(np.median(v)) if v else None) for cl, v in per.items()},
       'mean_share_of_proposal_web_matched_by_class': mean_prop, 'seconds': time.perf_counter() - t0}
json.dump(out, open('results/r13_wiki.json' if K == 41 else f'results/r13_wiki_{K}shards.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
