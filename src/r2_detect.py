"""R2 open zero-shot detector scores (PREREG_R2.md). Needs results/r2_passages.pkl from r2_label.py.
Usage: python -I src/r2_detect.py DATA_DIR  -> results/r2_scores.pkl, results/r2_proposal_scores.pkl"""
import sys, os, re, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
D = sys.argv[1]
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

def binoculars(lo, lp_, y):  # Hans et al. 2024: log-ppl(performer) / cross-ppl(observer -> performer)
    ppl = torch.nn.functional.cross_entropy(lp_[0], y[0]).item()
    xppl = (-(torch.softmax(lo, -1) * torch.log_softmax(lp_, -1)).sum(-1)).mean().item()
    return ppl / xppl

def run(df, maxlen, tag):
    tk = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B'); m = load('Qwen/Qwen2.5-1.5B'); out = []; t0 = time.time()
    for i, s in enumerate(df.text):
        y, lg = logits(m, tk, s, maxlen); out.append(single(lg, y))
        if i % 100 == 0: print(tag, 'D1/D2', i, len(df), round(time.time() - t0), flush=True)
    df['D1_ll'], df['D2_fdg'] = zip(*out); del m
    tk0 = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); mo = load('Qwen/Qwen2.5-0.5B'); mp = load('Qwen/Qwen2.5-0.5B-Instruct'); b = []
    for i, s in enumerate(df.text):
        y, lo = logits(mo, tk0, s, maxlen); _, lp_ = logits(mp, tk0, s, maxlen); b.append(binoculars(lo, lp_, y))
        if i % 100 == 0: print(tag, 'D3', i, len(df), round(time.time() - t0), flush=True)
    df['D3_bino'] = b; df['ntok_model'] = [min(maxlen, len(tk(s).input_ids)) for s in df.text]
    return df

run(R, 256, 'passages').to_pickle('results/r2_scores.pkl')
run(props, 512, 'proposals').to_pickle('results/r2_proposal_scores.pkl')
print('done')
