"""Fig. 3 (R12): risk-coverage curves for the decision "chatbot text" on the planted-provenance benchmark, with a
stated confidence for every method and earliest plausible attribution with and without abstention by entry size
(13 specified routes, left; with the routes outside the specification, right). From results/r12_benchmark.json."""
import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
b = json.load(open('results/r12_benchmark.json'))
lab = {'ours_token': 'Attribution (abstains on X)', 'earliest_plausible_iota': 'Earliest plausible + abstention by entry size',
       'earliest_plausible': 'Earliest plausible attribution', 'dialogue_only': 'Dialogue only', 'naive_overlap': 'Naive 8-gram overlap'}
col = {'ours_token': '#1baf7a', 'earliest_plausible_iota': '#2a78d6', 'earliest_plausible': '#2a78d6', 'dialogue_only': '#eb6834', 'naive_overlap': '#7b5cc4'}
ls = {'earliest_plausible_iota': '--'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for ax, tag, title in ((axs[0], 'base', '(a) 13 specified routes'), (axs[1], 'ext', '(b) with 7 routes outside the specification')):
    rc = b['B10_selective_' + tag]
    for k in lab:
        cv = rc[k]['curve']; ax.plot([x for x, _ in cv], [y for _, y in cv], color=col[k], lw=1.4, ls=ls.get(k, '-'), label=f"{lab[k]} (AURC {rc[k]['aurc']:.3f})")
    c0 = rc['ours_token']['coverage_before_abstention']; ax.axvline(c0, color='#8a8984', lw=0.7, ls=':')
    ax.text(c0 + 0.01, 0.003, 'X begins', fontsize=6.5, color='#52514e')
    ax.set_title(title, loc='left', fontsize=8); ax.set_xlabel('Coverage (share of tokens decided)'); ax.set_xlim(0, 1); ax.set_ylim(0, 0.23)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
axs[0].set_ylabel('Risk (error among decided tokens)')
fig.tight_layout(); fig.savefig('results/fig3_risk_coverage_r12.png', dpi=300); print('ok')
