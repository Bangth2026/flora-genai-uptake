"""Plan R3 F-d: Binoculars with observer/performer roles swapped in both terms (paper notation; released code is used for D3).
Usage: python -I src/r3_d3swap.py -> results/r3_d3swap.json"""
import os, json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
torch.set_num_threads(os.cpu_count())
R = pd.read_pickle('results/r3_flora_paragraphs.pkl'); T = pd.read_pickle('results/r2_scores.pkl')[['user_id', 'j', 'text']]
R = R.merge(T, on=['user_id', 'j'])
tk = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
mo = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.float32).eval()
mp = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', dtype=torch.float32).eval()
@torch.no_grad()
def swap(s):
    ids = tk(s, return_tensors='pt', truncation=True, max_length=256).input_ids; y = ids[0, 1:]
    lo, lp = mo(ids).logits[0, :-1].float(), mp(ids).logits[0, :-1].float()
    lso = torch.log_softmax(lo, -1); ppl = -lso.gather(1, y[:, None]).mean()
    x = (-(torch.softmax(lp, -1) * lso).sum(-1)).mean()
    return (ppl / x).item()
R['D3s'] = [-swap(s) for s in R.text]
out = {}
for a, b in [('chatbot', 'composed_all'), ('chatbot', 'typed'), ('chatbot', 'untraced'), ('untraced', 'typed')]:
    m = R.cls3.isin([a, b]) if b != 'composed_all' else R.label.isin(['chatbot', 'composed'])
    yy = (R[m].cls3 == a) if b != 'composed_all' else (R[m].label == 'chatbot')
    out[f'{a}_vs_{b}'] = float(roc_auc_score(yy, R[m].D3s))
c = R[R.label == 'composed']; out['l1_diff_sd_composed'] = float((c[c.eng == 0].D3s.mean() - c[c.eng == 1].D3s.mean()) / c.D3s.std())
json.dump(out, open('results/r3_d3swap.json', 'w'), indent=1); print(out)
