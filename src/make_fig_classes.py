"""Fig. 4: provenance-class composition of the final artefacts (both datasets) and segment uptake by human response-use code."""
import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ink, sub, grid = '#0b0b0b', '#52514e', '#e4e3df'
col = {'C': '#1baf7a', 'E': '#eb6834', 'S': '#8a8984', 'T': '#2a78d6', 'X': '#7b5cc4'}
name = {'C': 'C chatbot-first', 'E': 'E echo', 'S': 'S shared source', 'T': 'T typed', 'X': 'X untraced insertion'}
f = {k: v['mean'] for k, v in json.load(open('results/r4_flora_stats.json'))['classes_edit'].items()}; r = json.load(open('results/r3_rs_stats.json'))
rs = {k: v['mean'] for k, v in json.load(open('results/r4_rs_stats.json'))['classes_edit'].items()}; seg = r['segments']['any_uptake_by_code']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={'width_ratios': [1.15, 1]})
for i, (lab, d) in enumerate([('FLoRA\n(writing, n = 275)', f), ('RelianceScope\n(code, n = 79)', rs)]):
    left = 0
    for k in 'CESTX':
        a.barh(i, d[k], left=left, color=col[k], height=0.55, edgecolor='white', lw=0.8)
        if d[k] >= .06: a.text(left + d[k] / 2, i, f'{d[k]*100:.0f}%', ha='center', va='center', color='white', fontsize=7, fontweight='bold')
        left += d[k]
a.set_yticks([0, 1]); a.set_yticklabels(['FLoRA\n(writing, n = 275)', 'RelianceScope\n(code, n = 79)']); a.invert_yaxis()
a.set_xlim(0, 1); a.set_xlabel('Mean share of final-artefact tokens'); a.xaxis.set_major_formatter(lambda v, p: f'{v:.0%}')
for s in ['top', 'right', 'left']: a.spines[s].set_visible(False)
a.tick_params(axis='y', length=0); a.legend([plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in 'CESTX'], [name[k] for k in 'CESTX'],
                                            ncol=3, fontsize=6.3, frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.28))
a.set_title('(a) Provenance classes', loc='left', fontsize=8, color=ink)
order = sorted([k for k in seg if seg[k][1] >= 8], key=lambda k: seg[k][0])
pos = {'copy the response', 'write over the response'}
b.barh(range(len(order)), [seg[k][0] for k in order], color=['#1baf7a' if k in pos else '#b9b8b3' for k in order], height=0.6)
b.set_yticks(range(len(order))); b.set_yticklabels([f'{k} ({seg[k][1]})' for k in order], fontsize=6.6)
b.set_xlim(0, 1); b.xaxis.set_major_formatter(lambda v, p: f'{v:.0%}'); b.set_xlabel('Segments with any chatbot-first uptake')
for i, k in enumerate(order): b.text(seg[k][0] + .02, i, f'{seg[k][0]:.0%}', va='center', fontsize=6.6, color=sub)
for s in ['top', 'right', 'left']: b.spines[s].set_visible(False)
b.tick_params(axis='y', length=0); b.grid(axis='x', color=grid, lw=.6); b.set_axisbelow(True)
b.set_title('(b) Segments by human response-use code', loc='left', fontsize=8, color=ink)
fig.tight_layout(); fig.savefig('results/fig4_classes.png', dpi=300); print('ok')
