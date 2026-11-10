"""R2 open zero-shot detector scores (PREREG_R2.md). Needs results/r2_passages.pkl from r2_label.py.
Usage: python -I src/r2_detect.py DATA_DIR STAGE (d12 | d3 | merge), one process per stage to fit in memory -> results/r2_scores.pkl, results/r2_proposal_scores.pkl"""
import sys, os, re, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
D, STAGE = sys.argv[1], sys.argv[2]
R = pd.read_pickle('results/r2_passages.pkl'); R = R[R.label != 'mixed'].reset_index(drop=True)
p = pd.read_excel(f'{D}/proposal.xlsx').drop_duplicates('user_id').set_index('user_id')
users = sorted(set(R.user_id) | set(pd.read_excel(f'{D}/dialogue.xlsx').user_id) & set(p.index))
props = pd.DataFrame({'user_id': users, 'text': [str(p.loc[u, 'final_proposal']) for u in users]})
DT = {'Qwen/Qwen2.5-1.5B': torch.bfloat16}  # deviation R2-1: 1.5B weights in bfloat16 (float32 ran out of 7 GB RAM)
load = lambda m: AutoModelForCausalLM.from_pretrained(m, dtype=DT.get(m, torch.float32)).eval()

@torch.no_grad()
def logits(model, tk, text, maxlen):
    ids = tk(text, return_tensors='pt', truncation=True, max_length=maxlen).input_ids
    return ids[:, 1:], model(ids).logits[:, :-1].float()

def single(lg, y):  # D1 mean log-likelihood, D2 Fast-DetectGPT analytic (sampling = scoring)
    lp = torch.log_softmax(lg, -1); ll = lp.gather(-1, y[..., None]).squeeze(-1)
    pr = lp.exp(); mu = (pr * lp).sum(-1); var = (pr * lp ** 2).sum(-1) - mu ** 2
    return ll.mean().item(), ((ll.sum() - mu.sum()) / var.sum().sqrt()).item()

def binoculars(lo, lp_, y, ch=64):  # Hans et al. 2024: log-ppl(performer) / cross-ppl(observer -> performer); chunked to save memory
    ppl = torch.nn.functional.cross_entropy(lp_[0], y[0]).item(); x = 0.0
    for a in range(0, lo.shape[1], ch):
        x += (-(torch.softmax(lo[0, a:a + ch], -1) * torch.log_softmax(lp_[0, a:a + ch], -1)).sum(-1)).sum().item()
    return ppl / (x / lo.shape[1])

def d12(df, maxlen, tag):
    tk = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B'); m = load('Qwen/Qwen2.5-1.5B'); out = []; t0 = time.time()
    for i, s in enumerate(df.text):
        y, lg = logits(m, tk, s, maxlen); out.append(single(lg, y))
        if i % 100 == 0: print(tag, 'D1/D2', i, len(df), round(time.time() - t0), flush=True)
    o = pd.DataFrame(out, columns=['D1_ll', 'D2_fdg']); o['ntok_model'] = [min(maxlen, len(tk(s).input_ids)) for s in df.text]
    o.to_pickle(f'results/_ck_{tag}_d12.pkl')

def d3(df, maxlen, tag):
    tk0 = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); mo = load('Qwen/Qwen2.5-0.5B'); mp = load('Qwen/Qwen2.5-0.5B-Instruct'); b = []; t0 = time.time()
    for i, s in enumerate(df.text):
        y, lo = logits(mo, tk0, s, maxlen); _, lp_ = logits(mp, tk0, s, maxlen); b.append(binoculars(lo, lp_, y))
        if i % 100 == 0: print(tag, 'D3', i, len(df), round(time.time() - t0), flush=True)
    pd.DataFrame({'D3_bino': b}).to_pickle(f'results/_ck_{tag}_d3.pkl')

for df, ml, tag in [(R, 256, 'passages'), (props, 512, 'proposals')]:
    done = lambda st: os.path.exists(f'results/_ck_{tag}_{st}.pkl')
    if STAGE == 'd12' and not done('d12'): d12(df, ml, tag)
    elif STAGE == 'd3' and not done('d3'): d3(df, ml, tag)
    elif STAGE == 'merge':
        out = pd.concat([df, pd.read_pickle(f'results/_ck_{tag}_d12.pkl'), pd.read_pickle(f'results/_ck_{tag}_d3.pkl')], axis=1)
        out.to_pickle('results/r2_scores.pkl' if tag == 'passages' else 'results/r2_proposal_scores.pkl')
print('done', STAGE)
