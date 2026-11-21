"""Analysis log R19: Fig. 4 (share of students at or above each level of chatbot text under naive overlap, removal of
all prompt 8-grams, U, and U with large-entry echo or X-with-C added; results/r19_flora.json) and Fig. 5 (FLoRA class
composition and the subtypes of X; results/r11_flora.json, results/r17_flora_xsub.json) of the article.
Usage: python -I src/r19_make_figs.py DATA_DIR OUT_DIR (run from the repository root)"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src'); import run_s1 as R
D, OUT = sys.argv[1], sys.argv[2]
d = pd.read_excel(f'{D}/dialogue.xlsx'); p = pd.read_excel(f'{D}/proposal.xlsx').set_index('user_id')
users = sorted(set(d.user_id) & set(p.index))
naive = np.array([R.coverage(R.tok(p.loc[u, 'final_proposal']), R.ngrams(R.tok(' '.join(d[d.user_id == u].chatbot_utterance_text.astype(str))), 8), 8) for u in users])
blind = np.load('results/Ue.npy'); dist = json.load(open('results/r19_flora.json'))['F40_distributions']
ink, sub, grid = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': sub, 'xtick.color': sub, 'ytick.color': sub,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6.4, 3.5))
series = [(naive, 'Naive overlap', '#2a78d6', '-'), (blind, 'All prompt 8-grams removed', '#8a8984', '--'),
          (np.array(dist['U']), 'U (chatbot-first)', '#1baf7a', '-'), (np.array(dist['U_plus_large_echo']), 'U + large-entry echo', '#eb6834', '-.'),
          (np.array(dist['U_plus_X_with_C']), 'U + X-with-C', '#7b5cc4', ':')]
for v, lab, c, ls in series:
    x = np.sort(v); ax.step(x, 1 - np.arange(len(x)) / len(x), where='post', color=c, lw=1.8, ls=ls, label=lab)
for t in (.25, .5, .75): ax.axvline(t, color=grid, lw=0.8, zorder=0)
ax.set_xlabel('Share of the proposal (x)'); ax.set_ylabel('Share of students at or above x'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.xaxis.set_major_formatter(lambda v, q: f'{v:.0%}'); ax.yaxis.set_major_formatter(lambda v, q: f'{v:.0%}')
ax.legend(frameon=False, loc='upper right', fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_levels_r19.png', dpi=300); plt.close(fig)

col = {'C': '#1baf7a', 'E': '#eb6834', 'S': '#8a8984', 'T': '#2a78d6', 'X': '#7b5cc4'}
name = {'C': 'C chatbot-first', 'E': 'E echo', 'S': 'S shared', 'T': 'T typed', 'X': 'X unresolved'}
f = json.load(open('results/r11_flora.json'))['F1_shares_token_entry']
xs = json.load(open('results/r17_flora_xsub.json'))['F29_x_subtypes_without_embedding_flags']['mean_share_of_proposal']
order = ['X-with-C', 'X-first-snapshot', 'X-reentered', 'X-cohort-reply', 'X-log-gap', 'X-no-evidence']
shade = ['#5a3fa6', '#7b5cc4', '#9a83d6', '#b7a6e2', '#d3c8ee', '#e9e3f7']
fig, ax = plt.subplots(figsize=(6.4, 2.4))
left = 0
for k in 'CESTX':
    ax.barh(1, f[k], left=left, color=col[k], height=0.5, edgecolor='white', lw=0.8)
    if f[k] >= .06: ax.text(left + f[k] / 2, 1, f'{name[k]}\n{f[k]*100:.1f}%', ha='center', va='center', color='white', fontsize=6.5)
    left += f[k]
scale = f['X'] / sum(xs[k] for k in order); left = 1 - f['X']
for k, c in zip(order, shade):
    w = xs[k] * scale
    ax.barh(0, w, left=left, color=c, height=0.5, edgecolor='white', lw=0.8)
    if w >= .05: ax.text(left + w / 2, 0, {'X-with-C': 'with C', 'X-first-snapshot': 'first\nsnapshot', 'X-reentered': 're-\nentered', 'X-no-evidence': 'no evidence'}.get(k, k) + f'\n{xs[k]*100:.1f}%', ha='center', va='center', color='white' if k in order[:2] else ink, fontsize=6.0)
    left += w
ax.plot([1 - f['X'], 1 - f['X']], [0.25, 0.75], color=sub, lw=0.6, ls=':')
ax.set_yticks([0, 1]); ax.set_yticklabels(['X by subtype', 'All classes']); ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(lambda v, q: f'{v:.0%}'); ax.set_xlabel('Mean share of the FLoRA proposal (275 students)')
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig5_classes_r19.png', dpi=300); plt.close(fig); print('ok')
