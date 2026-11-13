"""R11 version (notation U). Pipeline diagram (Fig. 2 of the manuscript): four evidential classes and the unresolved outcome."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ink, sub, blue, orange, aqua, grey, purple = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#7b5cc4'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, ax = plt.subplots(figsize=(7.4, 4.6)); ax.set_xlim(-2, 104); ax.set_ylim(-14, 58); ax.axis('off')
def box(x, y, w, h, title, body, c, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc='white', ec=c, lw=lw))
    ax.text(x + w / 2, y + h - 2.0, title, ha='center', va='top', color=ink, fontweight='bold', fontsize=7.6)
    ax.text(x + w / 2, y + h - 5.4, body, ha='center', va='top', color=sub, fontsize=6.6, linespacing=1.3)
def arr(x1, y1, x2, y2):
    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color=grey, lw=1.1, shrinkA=2, shrinkB=2))
ax.text(9, 56.5, 'Logged inputs', ha='center', color=sub, fontsize=7, style='italic')
ins = [('Final artefact A', 'proposal or code'), ('Chatbot replies R', 'text + time'), ('Student turns Q', 'text (+ code) + time'),
       ('Editor snapshots W', 'time-stamped states'), ('Shared sources', 'readings, template, peers')]
for i, (t, b) in enumerate(ins): box(0, 44 - i * 12.5, 18, 10, t, b, grey)
ax.text(51, 56.5, 'Attribution of each n-gram of A (applied in order)', ha='center', color=sub, fontsize=7, style='italic')
st = [('1 Direction', 'in R, but in Q or W\nbefore the reply → E', orange),
      ('2 Shared source', 'in readings, template\nor ≥ k peers → S', orange),
      ('3 Order check', 'in R, reply no later than\nfirst editor state → C', aqua),
      ('4 Entry mode', 'small edit → T\nlarge insertion → X', purple)]
for i, (t, b, c) in enumerate(st): box(25, 42 - i * 13.5, 26, 11, t, b, c)
for i in range(3): arr(38, 42 - i * 13.5, 38, 42 - (i + 1) * 13.5 + 11)
for i in range(5): arr(18, 49 - i * 12.5, 25, 47.5 - min(i, 3) * 13.5)
box(57, 13, 19, 36, 'Evidential classes', 'C  chatbot-first\n\nE  echo of the\n    student\'s text\n\nS  shared source\n\nT  typed in editor\n\nUnresolved outcome\nX  source not in logs', ink, lw=2)
for i in range(4): arr(51, 47.5 - i * 13.5, 57, 31)
box(82, 36, 21, 13, 'Uptake U', 'share of A in class C\n(offline: Algorithms 1–2;\nonline: Algorithm 3)', aqua)
box(82, 17, 21, 15, 'Validation', 'planted provenance,\npastes, keystrokes,\nchance null, human codes', blue)
box(82, -2, 21, 15, 'Uses', 'feedback at paste time,\nenvironment design;\noutcomes, detectors (Suppl. C)', grey)
arr(76, 40, 82, 42); arr(76, 28, 82, 24); arr(76, 18, 82, 6)
box(25, -13, 51, 8, 'Clock check (Algorithm S2)', 'chat-box Enter keystrokes vs ask times → offset δ', grey, lw=1.2)
arr(38, -5, 38, 1.5)
fig.tight_layout(); fig.savefig('results/fig0_pipeline_r11.png', dpi=300); print('ok')
