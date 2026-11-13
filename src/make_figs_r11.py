"""R11 version (notation U; CCDF only). Figures for the manuscript from saved results. Usage: python -I src/make_figs.py DATA_DIR"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src'); import run_s1 as R
D = sys.argv[1]
d = pd.read_excel(f'{D}/dialogue.xlsx'); p = pd.read_excel(f'{D}/proposal.xlsx').set_index('user_id')
users = sorted(set(d.user_id) & set(p.index))
U = np.array([R.coverage(R.tok(p.loc[u, 'final_proposal']), R.ngrams(R.tok(' '.join(d[d.user_id == u].chatbot_utterance_text.astype(str))), 8), 8) for u in users])
Ue = np.load('results/Ue.npy'); Up = np.load('results/U_R1.npy')
score = p.reindex(users).proposal_score.values
C = ['#2a78d6', '#eb6834', '#1baf7a']; ink = '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.edgecolor': ink, 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': ink, 'ytick.color': ink, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for v, lab, c, ls in [(U, 'Naive overlap', C[0], '-'), (Ue, 'Direction-blind echo removal', C[1], '--'), (Up, 'Provenance-checked U (primary)', C[2], ':')]:
    x = np.sort(v); ax.step(x, 1 - np.arange(len(x)) / len(x), where='post', color=c, lw=2, ls=ls, label=lab)
ax.set_xlabel('Share of proposal tokens covered by chatbot 8-grams'); ax.set_ylabel('Share of students at or above x')
ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.legend(frameon=False, loc='upper right'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig('results/fig1_uptake_ccdf_r11.png', dpi=300)
fig, ax = plt.subplots(figsize=(6.2, 3.4))
from statsmodels.nonparametric.smoothers_lowess import lowess
hi = Up >= .5; jit = np.random.default_rng(1).uniform(-.1, .1, len(score))
ax.scatter(Up[~hi], score[~hi] + jit[~hi], s=20, color=C[0], alpha=.65, edgecolor='#fcfcfb', lw=.6, label='U* < 0.5')
ax.scatter(Up[hi], score[hi] + jit[hi], s=24, marker='D', color=C[1], alpha=.8, edgecolor='#fcfcfb', lw=.6, label='U* ≥ 0.5')
grid = np.linspace(0, 1, 50); rng = np.random.default_rng(7); curves = []
for _ in range(500):
    b = rng.choice(len(Up), len(Up)); f = lowess(score[b], Up[b], frac=.6, return_sorted=True)
    curves.append(np.interp(grid, f[:, 0], f[:, 1]))
f = lowess(score, Up, frac=.6, return_sorted=True)
ax.fill_between(grid, np.percentile(curves, 2.5, 0), np.percentile(curves, 97.5, 0), color=ink, alpha=.12, lw=0)
ax.plot(f[:, 0], f[:, 1], color=ink, lw=2, label='LOWESS with 95% band')
ax.set_xlabel('Provenance-checked uptake U*'); ax.set_ylabel('Proposal score (0–15)')
ax.grid(color='#e6e5e0', lw=0.6); ax.set_xlim(-0.02, 1.02); ax.legend(frameon=False, loc='lower right', fontsize=8)
fig.tight_layout(); pass
print('ok')
