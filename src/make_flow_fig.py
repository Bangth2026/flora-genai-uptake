"""Fig. 1 of the v6 manuscript: routes by which text can reach the editor, and the provenance class each route receives."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
col = {'C': '#1baf7a', 'E': '#eb6834', 'S': '#8a8984', 'T': '#2a78d6', 'X': '#7b5cc4', 'k': '#52514e'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(-2, 52); ax.axis('off')
def box(x, y, w, h, t):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.0', fc='#f3f2ee', ec=col['k'], lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8)
def arrow(x1, y1, x2, y2, c):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.5))
rows = {'Student': 44, 'Chatbot\n(logged replies)': 30, 'Shared sources\n(readings, template)': 16, 'Sources not\nin the logs': 2}
for t, y in rows.items(): box(2, y, 19, 7, t)
box(82, 2, 16, 49, 'Editor\n(submitted\nartefact)')
y = rows['Student']
arrow(21, y + 5, 82, y + 5, col['T']); ax.text(51, y + 6.3, 'T  entered in small edits (typed)', color=col['T'], ha='center', fontsize=7.5, fontweight='bold')
arrow(21, y + 2, 82, y + 2, col['E']); ax.text(51, y - 0.6, "E  student's text, later repeated by a reply", color=col['E'], ha='center', fontsize=7.5, fontweight='bold')
arrow(11.5, y, 11.5, rows['Chatbot\n(logged replies)'] + 7, col['k']); ax.text(13, 39.5, 'prompt', color=col['k'], fontsize=7, ha='left')
y = rows['Chatbot\n(logged replies)']
arrow(21, y + 3.5, 82, y + 3.5, col['C']); ax.text(51, y + 4.8, 'C  in a reply before it reached the editor', color=col['C'], ha='center', fontsize=7.5, fontweight='bold')
y = rows['Shared sources\n(readings, template)']
arrow(21, y + 3.5, 82, y + 3.5, col['S']); ax.text(51, y + 4.8, 'S  text available to all students', color=col['S'], ha='center', fontsize=7.5, fontweight='bold')
y = rows['Sources not\nin the logs']
arrow(21, y + 3.5, 82, y + 3.5, col['X']); ax.text(51, y + 4.8, 'X  large insertion; source not recorded (unresolved)', color=col['X'], ha='center', fontsize=7.5, fontweight='bold')
fig.tight_layout(); fig.savefig('results/fig_flow.png', dpi=300); print('ok')
