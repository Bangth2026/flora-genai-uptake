"""Deviation D1 (added after the primary run, not in ANALYSIS_LOG_S1.md).
Chatbot replies can echo text the student pasted into a prompt (e.g. a draft sent for feedback).
Overlap with such text is the student's own writing, not uptake. Recompute U excluding 8-grams
that also occur in the student's own prompts, then repeat RQ1-RQ3 with the corrected label."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'src')
import run_s1 as R
from scipy.stats import spearmanr
data, out = sys.argv[1], sys.argv[2]
d = pd.read_excel(f'{data}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{data}/proposal.xlsx').set_index('user_id')
users = sorted(set(d.user_id) & set(p.index))
res = {}
U, Ue, paste = [], [], []
for u in users:
    x = d[d.user_id == u]
    prop = R.tok(p.loc[u, 'final_proposal'])
    bot = R.ngrams(R.tok(' '.join(x.chatbot_utterance_text.astype(str))), 8)
    stu = R.ngrams(R.tok(' '.join(x.user_utterance_text.astype(str))), 8)
    U.append(R.coverage(prop, bot, 8)); Ue.append(R.coverage(prop, bot - stu, 8))
    paste.append(R.coverage(prop, stu, 8))
U, Ue, paste = map(np.array, (U, Ue, paste))
q = lambda v: {k: float(t) for k, t in zip(['p25', 'median', 'p75', 'p90'], np.percentile(v, [25, 50, 75, 90]))}
res['U8'] = q(U); res['U8_echo_removed'] = q(Ue); res['proposal_covered_by_own_prompts'] = q(paste)
res['spearman_U_Ue'] = float(spearmanr(U, Ue)[0])
res['share_Ue_ge_0.25'] = float((Ue >= .25).mean()); res['share_Ue_ge_0.50'] = float((Ue >= .5).mean())
np.save('results/Ue.npy', Ue)
json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=1))
