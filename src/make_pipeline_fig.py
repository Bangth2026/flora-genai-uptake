"""Pipeline diagram (Fig. 1 of the manuscript)."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ink, sub, blue, orange, aqua, grey = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.set_xlim(-2, 102); ax.set_ylim(-7, 57); ax.axis('off')
def box(x, y, w, h, title, body, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc='white', ec=c, lw=1.6))
    ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', color=ink, fontweight='bold')
    ax.text(x + w / 2, y + h - 5.6, body, ha='center', va='top', color=sub, fontsize=7, linespacing=1.3)
def arr(x1, y1, x2, y2, label=None):
    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color=grey, lw=1.2, shrinkA=2, shrinkB=2))
    if label: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, label, ha='center', color=sub, fontsize=6.5, style='italic')
# inputs
box(0, 42, 19, 13, 'Final proposal P', 'tokens p1..pm', grey)
box(0, 26, 19, 13, 'Chatbot replies R', 'text + time of\neach reply', grey)
box(0, 10, 19, 13, 'Student prompts Q', 'text of each\nstudent turn', grey)
box(0, -4, 19, 11, 'Writing log W', 'time-stamped\neditor snapshots', grey)
# measures
box(26, 40, 21, 15, 'Naive uptake U', 'C8(P, R)\n8-gram containment', blue)
box(26, 20, 21, 15, 'Echo-corrected Ue', 'C8(P, R \\ Q)\nAlgorithm 1', orange)
box(53, 20, 22, 15, 'Provenance check', 'chatbot-first if\nt_reply ≤ t_editor + δ\nAlgorithm 2', aqua)
box(53, -4, 22, 15, 'Clock offset δ', 'from prompt text already\nin editor (no outcome)\nAlgorithm 3', grey)
box(80, 20, 20, 15, 'U_prov (primary)', 'share of P covered by\nchatbot-first 8-grams', aqua)
box(80, 40, 20, 15, 'RQ2–RQ4', 'detection (Algorithm 4),\ngrades, first language', ink)
for y in (48.5, 32.5): arr(19, y, 26, 47.5)
arr(19, 32.5, 26, 27.5); arr(19, 16.5, 26, 27.5, 'remove'); arr(47, 27.5, 53, 27.5); arr(19, 46, 26, 31)
arr(19, 1.5, 53, 3.5); arr(64, 11, 64, 20); arr(19, 1.5, 53, 22)
arr(75, 27.5, 80, 27.5); arr(90, 35, 90, 40)
ax.text(64, 37, 'rule fixed in advance:\n≥ 80% chatbot-first → Ue, else U_prov', ha='center', color=sub, fontsize=6.5)
fig.tight_layout(); fig.savefig('results/fig0_pipeline.png', dpi=300); print('ok')
