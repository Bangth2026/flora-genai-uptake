"""Analysis log R18, figure of a worked example (Fig. 6 of the article): the synthetic student of demo/ with each passage
coloured by its provenance class and the log events that witness the label. Uses demo/provenance.py only; reproduces
no study result. Usage: python -I src/r18_make_fig_example.py OUT.png"""
import sys, os, json, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'demo'))
import provenance as P
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
L = json.load(open(os.path.join(HERE, '..', 'demo', 'synthetic_logs.json')))
n, tau = L['n'], L['tau']
A, cls, lab, U = P.attribute(L['final'], L['replies'], L['turns'], L['snapshots'], L['shared'], n, tau)
G = P.ngrams(A, n); t_ed, ins = P.editor_entry(G, L['snapshots'], n)
t_bot = P.first_times(L['replies'], n); t_pr = P.first_times(L['turns'], n)
runs = []; i = 0
while i < len(A):
    j = i
    while j < len(A) and cls[j] == cls[i]: j += 1
    runs.append((cls[i], i, j)); i = j
def fmt(t): return f'{t:g}'
wit = []
for c, i, j in runs:
    gs = [g for k, g in enumerate(G) if i <= k < j and k + n <= j] or [G[min(i, len(G) - 1)]]
    te = min(t_ed[g] for g in gs if g in t_ed); sz = max(ins[g] for g in gs if g in t_ed)
    if c == 'C':
        tb = min(t_bot[g] for g in gs if g in t_bot)
        w = f'Reply at t\u00a0=\u00a0{fmt(tb)} contains it; first in the editor at t\u00a0=\u00a0{fmt(te)}, in an insertion of {sz} tokens; in no earlier turn'
    elif c == 'E':
        tp = min(t_pr[g] for g in gs if g in t_pr); tb = min(t_bot[g] for g in gs if g in t_bot)
        w = f'Writer turn at t\u00a0=\u00a0{fmt(tp)} contains it; the reply at t\u00a0=\u00a0{fmt(tb)} repeats it (R1, turn first)'
    elif c == 'S':
        w = f'Its 8-grams occur in a shared reading (R3); entered at t\u00a0=\u00a0{fmt(te)} in an edit of {sz} token{"s" if sz != 1 else ""}'
    elif c == 'T':
        w = f'Entered in edits of at most {tau} tokens (largest {sz}); no reply, turn or shared text matches'
    else:
        w = f'Entered at t\u00a0=\u00a0{fmt(te)} in one insertion of {sz} tokens (> τ = {tau}); no reply, turn or shared text matches'
    wit.append(w)
col = {'C': '#1baf7a', 'E': '#eb6834', 'S': '#8a8984', 'T': '#2a78d6', 'X': '#7b5cc4'}
name = {'C': 'C chatbot-first', 'E': 'E echo', 'S': 'S shared source', 'T': 'T typed', 'X': 'X unresolved'}
ink, sub = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})
fig, ax = plt.subplots(figsize=(7.4, 4.3)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.0, 0.985, 'Final text (passages in order)', fontsize=8, fontweight='bold', color=ink, va='top')
ax.text(0.56, 0.985, 'Log events that witness the label', fontsize=8, fontweight='bold', color=ink, va='top')
y = 0.93; H = 0.165
for (c, i, j), w in zip(runs, wit):
    ax.add_patch(FancyBboxPatch((0.0, y - H + 0.012), 0.535, H - 0.024, boxstyle='round,pad=0.004,rounding_size=0.008', fc=col[c] + '22', ec=col[c], lw=1.2))
    ax.text(0.012, y - 0.012, name[c], fontsize=7.5, fontweight='bold', color=col[c], va='top')
    ax.text(0.012, y - 0.045, '\n'.join(textwrap.wrap(' '.join(A[i:j]), 57)), fontsize=6.8, color=ink, va='top', linespacing=1.25)
    ax.text(0.56, y - 0.02, '\n'.join(textwrap.wrap(w, 60)), fontsize=6.8, color=sub, va='top', linespacing=1.3)
    y -= H
ax.text(0.0, 0.07, f'Uptake U = share of tokens in C = {U:.3f}.  Times in minutes from the first snapshot; n = {n}, τ = {tau}, δ = 0.', fontsize=6.8, color=sub, va='top')
fig.savefig(sys.argv[1], dpi=300, bbox_inches='tight'); print('ok', [(c, j - i) for c, i, j in runs])
