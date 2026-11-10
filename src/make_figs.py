"""Figures for the manuscript from saved results. Usage: python -I src/make_figs.py DATA_DIR"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src'); import run_s1 as R
D = sys.argv[1]
d = pd.read_excel(f'{D}/dialogue.xlsx'); p = pd.read_excel(f'{D}/proposal.xlsx').set_index('user_id')
users = sorted(set(d.user_id) & set(p.index))
U = np.array([R.coverage(R.tok(p.loc[u, 'final_proposal']), R.ngrams(R.tok(' '.join(d[d.user_id == u].chatbot_utterance_text.astype(str))), 8), 8) for u in users])
Ue = np.load('results/Ue.npy'); Up = np.load('results/Uprov_primary_0h.npy')
score = p.reindex(users).proposal_score.values
C = ['#2a78d6', '#eb6834', '#1baf7a']; ink = '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.edgecolor': ink, 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': ink, 'ytick.color': ink, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for v, lab, c, ls in [(U, 'Naive overlap U', C[0], '-'), (Ue, 'Echo removed Ue', C[1], '--'), (Up, 'Provenance checked U_prov', C[2], ':')]:
    x = np.sort(v); ax.step(x, 1 - np.arange(len(x)) / len(x), where='post', color=c, lw=2, ls=ls, label=lab)
ax.set_xlabel('Share of proposal tokens covered by chatbot 8-grams'); ax.set_ylabel('Share of students at or above x')
ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.legend(frameon=False, loc='upper right'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig('results/fig1_uptake_ccdf.png', dpi=300)
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.scatter(Up, score + np.random.default_rng(1).uniform(-.1, .1, len(score)), s=22, color=C[0], alpha=.7, edgecolor='#fcfcfb', lw=.6)
ax.set_xlabel('Provenance-checked uptake U_prov'); ax.set_ylabel('Proposal score (0–15)')
ax.grid(color='#e6e5e0', lw=0.6); ax.set_xlim(-0.02, 1.02)
fig.tight_layout(); fig.savefig('results/fig2_score_vs_uptake.png', dpi=300)
print('ok')
