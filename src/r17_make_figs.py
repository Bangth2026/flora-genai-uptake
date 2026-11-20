"""Analysis log R17: Supplementary Fig. S1 (trade-off of n: benchmark recall of chatbot-first text against chance
coverage in FLoRA) and Fig. 3 of the article (risk--coverage in the standard form, results/r17_benchmark.json).
Usage: python -I src/r17_make_figs.py OUT_DIR"""
import sys, os, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1]; R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
b = json.load(open(f'{R}/r16_benchmark.json'))['B24_sensitivity']; f = json.load(open(f'{R}/r17_flora.json'))['F27_null_coverage_by_n']
ns = [5, 6, 8, 10, 12]
fig, ax = plt.subplots(figsize=(5.8, 2.6)); ax2 = ax.twinx()
ax.plot(ns, [b['base'][f'n{n}_tau5_delta0']['C_r'] for n in ns], 'o-', color='#1f4e79', label='Benchmark recall, 13 routes')
ax.plot(ns, [b['ext'][f'n{n}_tau5_delta0']['C_r'] for n in ns], 's--', color='#1f4e79', label='Benchmark recall, 20 routes')
nn = sorted(int(k) for k in f); ax2.plot(nn, [100 * f[str(n)]['null_mean'] for n in nn], '^-', color='#b4462b', label='FLoRA chance coverage (%)')
ax.axvline(8, color='grey', lw=.8, ls=':'); ax.set_xlabel('n (tokens per n-gram)'); ax.set_ylabel('Recall of chatbot-first text'); ax2.set_ylabel('Mean chance coverage (%)', color='#b4462b')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right', frameon=False)
fig.tight_layout(); fig.savefig(f'{OUT}/figS1_n_tradeoff.png', dpi=300); plt.close(fig)
p = f'{R}/r17_benchmark.json'
if os.path.exists(p):
    d = json.load(open(p))['B29_standard_selective']
    lab = {'ours_token': 'Attribution', 'earliest_plausible': 'Earliest plausible attribution', 'earliest_plausible_iota': 'Earliest plausible + abstention',
           'dialogue_only': 'Dialogue only', 'naive_overlap': 'Naive overlap', 'paste_log': 'Paste-log rule'}
    sty = {'ours_token': ('#1f4e79', '-'), 'earliest_plausible': ('#7f7f7f', '-'), 'earliest_plausible_iota': ('#7f7f7f', '--'),
           'dialogue_only': ('#b4462b', '-'), 'naive_overlap': ('#d9a441', '-'), 'paste_log': ('#5a9e6f', '-')}
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 3.1), sharey=True)
    for ax, (name, ttl) in zip(axs, (('base', '(a) 13 specified routes'), ('ext', '(b) with 7 routes outside the specification'))):
        for k in lab:
            c = d[name][k]['curve']; x = [q[0] for q in c][3:]; y = [100 * q[1] for q in c][3:]
            ax.plot(x, y, color=sty[k][0], ls=sty[k][1], lw=1.2, label=lab[k])
        ax.axvline(d[name]['common_coverage'], color='grey', lw=.6, ls=':')
        ax.set_title(ttl, fontsize=8); ax.set_xlabel('Coverage'); ax.set_xlim(0, 1); ax.set_ylim(0, 30)
    axs[0].set_ylabel('Selective risk (%)'); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, frameon=False, loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f'{OUT}/fig3_risk_coverage_standard.png', dpi=300); plt.close(fig)
print('ok')
